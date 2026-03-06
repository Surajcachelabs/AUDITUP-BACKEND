import {
  extractTranscriptSegments,
  isCsmSpeaker,
  loadDictionaryPhrases,
  normalizeText
} from './shared.js'
import { runOpenAiIntentFallback } from './openAiFallback.js'

const NOT_ATTORNEY_PATTERNS = [
  /not\s+(?:an\s+)?attorneys?/i,
  /not\s+lawyers?/i,
  /not\s+legal\s+attorneys?/i,
  /not\s+legal\s+representatives?/i,
  /cannot\s+provide\s+legal\s+representation/i
]

const NOT_LEGAL_ADVICE_PATTERNS = [
  /not\s+legal\s+advice/i,
  /does\s+not\s+constitute\s+legal\s+advice/i,
  /should\s+not\s+be\s+considered\s+legal\s+(?:advice|counsel)/i,
  /cannot\s+offer\s+legal\s+opinions/i,
  /informational\s+guidance\s*,?\s*not\s+legal\s+advice/i,
  /guidance\s+(?:rather\s+than|not)\s+legal\s+(?:advice|counsel)/i
]

const LEGAL_LIMITATION_CONTEXT_PATTERNS = [
  /we\s+provide\s+guidance\s+based\s+on\s+(?:our\s+)?(?:process|policy)/i,
  /you\s+may\s+consult\s+an\s+attorney/i,
  /this\s+call\s+is\s+for\s+informational\s+purposes\s+only/i,
  /nothing\s+discussed\s+(?:here|on\s+this\s+call)\s+is\s+legal\s+advice/i
]

function buildPassOutput({ timestamp, text }) {
  return {
    parameter: 'Legal Disclaimer',
    disclaimer_detected: true,
    score: 1,
    status: 'PASS',
    disclaimer_timestamp: timestamp,
    disclaimer_text: text
  }
}

function buildFailOutput() {
  return {
    parameter: 'Legal Disclaimer',
    disclaimer_detected: false,
    score: 0,
    status: 'FAIL',
    reason: 'No legal disclaimer provided by the CSM in the transcript.'
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

function hasLegalLimitationIntent(text) {
  const attorneySignal = NOT_ATTORNEY_PATTERNS.some((pattern) => pattern.test(text))
  const legalAdviceSignal = NOT_LEGAL_ADVICE_PATTERNS.some((pattern) => pattern.test(text))
  const contextSignal = LEGAL_LIMITATION_CONTEXT_PATTERNS.some((pattern) => pattern.test(text))

  return attorneySignal || legalAdviceSignal || contextSignal
}

function findRepositoryMatch(csmSegments, repositoryPhrases) {
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

  for (const segment of csmSegments) {
    const normalizedSegment = normalizeText(segment.text)

    for (const phraseRecord of phraseRecords) {
      const strictMatch =
        normalizedSegment.includes(phraseRecord.normalized) ||
        (phraseRecord.tokenCount === 1 &&
          normalizedSegment.split(' ').includes(phraseRecord.normalized))

      if (strictMatch && hasLegalLimitationIntent(segment.text)) {
        return {
          matched: true,
          timestamp: segment.timestamp,
          text: segment.text
        }
      }
    }
  }

  let bestCandidate = null

  for (const segment of csmSegments) {
    const normalizedSegment = normalizeText(segment.text)

    for (const phraseRecord of phraseRecords) {
      const phraseCoverage = computePhraseCoverage(normalizedSegment, phraseRecord.normalized)
      const ngramScore = jaccardSimilarity(normalizedSegment, phraseRecord.normalized)
      const score = Math.max(phraseCoverage, ngramScore)
      const threshold = phraseRecord.tokenCount <= 2 ? 0.8 : phraseRecord.tokenCount <= 5 ? 0.7 : 0.6

      if (score >= threshold && hasLegalLimitationIntent(segment.text)) {
        if (!bestCandidate || score > bestCandidate.score) {
          bestCandidate = {
            score,
            timestamp: segment.timestamp,
            text: segment.text,
            timestampSeconds: segment.timestampSeconds
          }
        }

        if (
          bestCandidate &&
          score === bestCandidate.score &&
          segment.timestampSeconds < bestCandidate.timestampSeconds
        ) {
          bestCandidate = {
            score,
            timestamp: segment.timestamp,
            text: segment.text,
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
      text: bestCandidate.text
    }
  }

  return {
    matched: false,
    timestamp: null,
    text: null
  }
}

export async function evaluateLegalDisclamer(transcriptPayload) {
  const segments = extractTranscriptSegments(transcriptPayload)
  const csmSegments = segments.filter((segment) => isCsmSpeaker(segment.speaker))

  if (csmSegments.length === 0) {
    return buildFailOutput()
  }

  const repositoryPhrases = await loadDictionaryPhrases([
    'legal_disclaimer_output.json',
    'LegalDisclaimer.json'
  ])
  const repositoryMatch = findRepositoryMatch(csmSegments, repositoryPhrases)

  if (repositoryMatch.matched) {
    return buildPassOutput({
      timestamp: repositoryMatch.timestamp,
      text: repositoryMatch.text
    })
  }

  const aiFallback = await runOpenAiIntentFallback({
    parameterName: 'Legal Disclaimer Detection',
    instructions:
      'Decide whether the CSM provides a legal disclaimer in these segments. A valid disclaimer means the CSM communicates they are not attorneys or that the discussion is not legal advice.',
    segments: csmSegments
  })

  if (aiFallback?.detected && hasLegalLimitationIntent(aiFallback.text ?? '')) {
    return buildPassOutput({
      timestamp: aiFallback.timestamp,
      text: aiFallback.text
    })
  }

  return buildFailOutput()
}
