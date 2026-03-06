import {
  extractTranscriptSegments,
  getTotalCallDurationSeconds,
  isCsmSpeaker,
  loadDictionaryPhrases,
  normalizeText
} from './shared.js'
import { runOpenAiIntentFallback } from './openAiFallback.js'

function buildPassOutput({ timestamp, closureText }) {
  return {
    parameter: 'Call Closure',
    closure_detected: true,
    score: 1,
    status: 'PASS',
    closure_timestamp: timestamp,
    closure_text: closureText,
    analysis_window_percentage: 20
  }
}

function buildFailOutput() {
  return {
    parameter: 'Call Closure',
    closure_detected: false,
    score: 0,
    status: 'FAIL',
    reason: 'No valid closure detected within the final 20% of the call duration'
  }
}

function buildNgramSet(text, n = 2) {
  const parts = normalizeText(text).split(' ').filter(Boolean)

  if (parts.length < n) {
    return new Set(parts)
  }

  const ngrams = new Set()
  for (let index = 0; index <= parts.length - n; index += 1) {
    ngrams.add(parts.slice(index, index + n).join(' '))
  }

  return ngrams
}

function jaccardSimilarity(left, right) {
  const leftSet = buildNgramSet(left, 2)
  const rightSet = buildNgramSet(right, 2)

  if (!leftSet.size || !rightSet.size) {
    return 0
  }

  let intersection = 0
  for (const item of leftSet) {
    if (rightSet.has(item)) {
      intersection += 1
    }
  }

  const union = leftSet.size + rightSet.size - intersection
  return union === 0 ? 0 : intersection / union
}

function buildTokenSet(value) {
  return new Set(normalizeText(value).split(' ').filter(Boolean))
}

function computePhraseCoverage(segmentText, phraseText) {
  const segmentTokens = buildTokenSet(segmentText)
  const phraseTokens = buildTokenSet(phraseText)

  if (phraseTokens.size === 0 || segmentTokens.size === 0) {
    return 0
  }

  let overlapCount = 0
  for (const token of phraseTokens) {
    if (segmentTokens.has(token)) {
      overlapCount += 1
    }
  }

  return overlapCount / phraseTokens.size
}

function evaluateRepositoryAndFlexibleMatch(segments, repositoryPhrases) {
  const phraseRecords = repositoryPhrases
    .map((phrase) => {
      const normalized = normalizeText(phrase)
      return {
        phrase,
        normalized,
        tokenCount: normalized.split(' ').filter(Boolean).length
      }
    })
    .filter((entry) => entry.normalized)

  for (let segmentIndex = segments.length - 1; segmentIndex >= 0; segmentIndex -= 1) {
    const segment = segments[segmentIndex]
    const normalizedSegment = normalizeText(segment.text)

    for (const phraseRecord of phraseRecords) {
      const strictMatch =
        normalizedSegment.includes(phraseRecord.normalized) ||
        (phraseRecord.tokenCount === 1 &&
          normalizedSegment.split(' ').includes(phraseRecord.normalized))

      if (strictMatch) {
        return {
          matched: true,
          timestamp: segment.timestamp,
          closureText: segment.text
        }
      }
    }
  }

  let bestCandidate = null

  for (const segment of segments) {
    const normalizedSegment = normalizeText(segment.text)

    for (const phraseRecord of phraseRecords) {
      const phraseCoverage = computePhraseCoverage(normalizedSegment, phraseRecord.normalized)
      const ngramScore = jaccardSimilarity(normalizedSegment, phraseRecord.normalized)
      const score = Math.max(phraseCoverage, ngramScore)
      const threshold = phraseRecord.tokenCount <= 2 ? 0.8 : phraseRecord.tokenCount <= 5 ? 0.7 : 0.6

      if (score >= threshold) {
        if (!bestCandidate || score > bestCandidate.score) {
          bestCandidate = {
            score,
            timestamp: segment.timestamp,
            closureText: segment.text,
            timestampSeconds: segment.timestampSeconds
          }
        }

        if (
          bestCandidate &&
          score === bestCandidate.score &&
          segment.timestampSeconds > bestCandidate.timestampSeconds
        ) {
          bestCandidate = {
            score,
            timestamp: segment.timestamp,
            closureText: segment.text,
            timestampSeconds: segment.timestampSeconds
          }
        }
      }
    }
  }

  if (bestCandidate) {
    return {
      matched: true,
      timestamp: bestCandidate.timestamp,
      closureText: bestCandidate.closureText
    }
  }

  return {
    matched: false,
    timestamp: null,
    closureText: null
  }
}

export async function evaluateClosingStatement(transcriptPayload) {
  const segments = extractTranscriptSegments(transcriptPayload)

  if (segments.length === 0) {
    return buildFailOutput()
  }

  const totalCallDurationSeconds = getTotalCallDurationSeconds(segments)
  const closureWindowStartSeconds = totalCallDurationSeconds * 0.8

  const csmSegmentsInClosureWindow = segments.filter(
    (segment) => isCsmSpeaker(segment.speaker) && segment.timestampSeconds >= closureWindowStartSeconds
  )

  if (csmSegmentsInClosureWindow.length === 0) {
    return buildFailOutput()
  }

  const closurePhrases = await loadDictionaryPhrases([
    'closures_output.json',
    'closing_statement_output.json'
  ])
  const phraseMatch = evaluateRepositoryAndFlexibleMatch(csmSegmentsInClosureWindow, closurePhrases)

  if (phraseMatch.matched) {
    return buildPassOutput({
      timestamp: phraseMatch.timestamp,
      closureText: phraseMatch.closureText
    })
  }

  const aiFallback = await runOpenAiIntentFallback({
    parameterName: 'Call Closure Detection',
    instructions:
      'Decide whether these CSM segments from the final 20% of the call contain a valid call-closing intent. Mark detected true only when the CSM is clearly ending the conversation.',
    segments: csmSegmentsInClosureWindow
  })

  if (aiFallback?.detected) {
    return buildPassOutput({
      timestamp: aiFallback.timestamp,
      closureText: aiFallback.text
    })
  }

  return buildFailOutput()
}
