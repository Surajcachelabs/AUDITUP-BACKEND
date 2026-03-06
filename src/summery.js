import {
  extractTranscriptSegments,
  getTotalCallDurationSeconds,
  isCsmSpeaker,
  loadDictionaryPhrases,
  normalizeText
} from './shared.js'
import { runOpenAiIntentFallback } from './openAiFallback.js'

const STOP_WORDS = new Set([
  'the',
  'a',
  'an',
  'and',
  'or',
  'to',
  'of',
  'in',
  'for',
  'on',
  'with',
  'we',
  'you',
  'your',
  'our',
  'is',
  'are',
  'was',
  'were',
  'be',
  'this',
  'that',
  'it',
  'as',
  'at',
  'from',
  'will',
  'today',
  'just',
  'let',
  'me',
  'so'
])

const INTENT_DEFINITIONS = [
  {
    name: 'Profile review discussion',
    keywords: ['profile', 'review', 'feedback', 'draft', 'assessment']
  },
  {
    name: 'Documentation clarification',
    keywords: ['document', 'documents', 'evidence', 'paperwork', 'files', 'clarify']
  },
  {
    name: 'Timeline inquiry',
    keywords: ['timeline', 'days', 'week', 'weeks', 'deadline', 'when', 'date']
  },
  {
    name: 'Process explanation',
    keywords: ['process', 'steps', 'workflow', 'procedure', 'phase', 'phases']
  }
]

function buildPassOutput({ timestamp, text }) {
  return {
    parameter: 'Call Summarisation',
    summary_detected: true,
    score: 1,
    status: 'PASS',
    summary_timestamp: timestamp,
    summary_text: text
  }
}

function buildFailOutput() {
  return {
    parameter: 'Call Summarisation',
    summary_detected: false,
    score: 0,
    status: 'FAIL',
    reason: 'No valid summary detected in the final 30% of the call.'
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

function detectCallIntent(segments) {
  const joinedText = normalizeText(segments.map((segment) => segment.text).join(' '))

  let bestIntent = null

  for (const intent of INTENT_DEFINITIONS) {
    let score = 0
    for (const keyword of intent.keywords) {
      if (joinedText.includes(keyword)) {
        score += 1
      }
    }

    if (!bestIntent || score > bestIntent.score) {
      bestIntent = {
        name: intent.name,
        keywords: intent.keywords,
        score
      }
    }
  }

  if (!bestIntent || bestIntent.score === 0) {
    return null
  }

  return bestIntent
}

function buildDiscussionKeywordSet(segments) {
  const tokenFrequency = new Map()

  for (const segment of segments) {
    const tokens = normalizeText(segment.text).split(' ').filter(Boolean)

    for (const token of tokens) {
      if (token.length < 3 || STOP_WORDS.has(token)) {
        continue
      }

      tokenFrequency.set(token, (tokenFrequency.get(token) ?? 0) + 1)
    }
  }

  const sorted = [...tokenFrequency.entries()].sort((a, b) => b[1] - a[1])
  return new Set(sorted.slice(0, 20).map(([token]) => token))
}

function isIntentAligned(summaryText, intent, discussionKeywordSet) {
  const normalizedSummary = normalizeText(summaryText)

  if (intent) {
    const intentKeywordMatch = intent.keywords.some((keyword) => normalizedSummary.includes(keyword))
    if (intentKeywordMatch) {
      return true
    }
  }

  const summaryTokens = normalizeText(summaryText).split(' ').filter(Boolean)
  let overlapCount = 0

  for (const token of summaryTokens) {
    if (discussionKeywordSet.has(token)) {
      overlapCount += 1
    }
  }

  return overlapCount >= 2
}

function selectBestSummaryCandidate(candidates, intent, discussionKeywordSet) {
  if (candidates.length === 0) {
    return null
  }

  const intentAligned = candidates.filter((candidate) =>
    isIntentAligned(candidate.text, intent, discussionKeywordSet)
  )

  if (intentAligned.length === 0) {
    return null
  }

  return intentAligned.reduce((latest, current) =>
    current.timestampSeconds > latest.timestampSeconds ? current : latest
  )
}

function findRepositoryOrFlexibleCandidates(segments, repositoryPhrases) {
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

  const candidates = []

  for (const segment of segments) {
    const normalizedSegment = normalizeText(segment.text)

    for (const phraseRecord of phraseRecords) {
      const strictMatch =
        normalizedSegment.includes(phraseRecord.normalized) ||
        (phraseRecord.tokenCount === 1 &&
          normalizedSegment.split(' ').includes(phraseRecord.normalized))

      if (strictMatch) {
        candidates.push(segment)
        break
      }

      const phraseCoverage = computePhraseCoverage(normalizedSegment, phraseRecord.normalized)
      const ngramScore = jaccardSimilarity(normalizedSegment, phraseRecord.normalized)
      const score = Math.max(phraseCoverage, ngramScore)
      const threshold = phraseRecord.tokenCount <= 2 ? 0.8 : phraseRecord.tokenCount <= 5 ? 0.7 : 0.6

      if (score >= threshold) {
        candidates.push(segment)
        break
      }
    }
  }

  return candidates
}

export async function evaluateSummery(transcriptPayload) {
  const segments = extractTranscriptSegments(transcriptPayload)

  if (segments.length === 0) {
    return buildFailOutput()
  }

  const totalCallDurationSeconds = getTotalCallDurationSeconds(segments)
  const summaryWindowStartSeconds = totalCallDurationSeconds * 0.7

  const csmWindowSegments = segments.filter(
    (segment) => isCsmSpeaker(segment.speaker) && segment.timestampSeconds >= summaryWindowStartSeconds
  )

  if (csmWindowSegments.length === 0) {
    return buildFailOutput()
  }

  const discussionSegmentsBeforeWindow = segments.filter(
    (segment) => segment.timestampSeconds < summaryWindowStartSeconds
  )
  const intentSourceSegments =
    discussionSegmentsBeforeWindow.length > 0 ? discussionSegmentsBeforeWindow : segments
  const detectedIntent = detectCallIntent(intentSourceSegments)
  const discussionKeywordSet = buildDiscussionKeywordSet(intentSourceSegments)

  const repositoryPhrases = await loadDictionaryPhrases([
    'summary_output.json',
    'summarization_output.json'
  ])
  const repositoryOrFlexibleCandidates = findRepositoryOrFlexibleCandidates(
    csmWindowSegments,
    repositoryPhrases
  )
  const repositoryOrFlexibleMatch = selectBestSummaryCandidate(
    repositoryOrFlexibleCandidates,
    detectedIntent,
    discussionKeywordSet
  )

  if (repositoryOrFlexibleMatch) {
    return buildPassOutput({
      timestamp: repositoryOrFlexibleMatch.timestamp,
      text: repositoryOrFlexibleMatch.text
    })
  }

  const aiFallback = await runOpenAiIntentFallback({
    parameterName: 'Call Summarisation Detection',
    instructions:
      'Decide whether these CSM segments from the final 30% of the call contain a valid summary that recaps discussion and outcomes. Mark detected true only if summary intent is clear.',
    segments: csmWindowSegments
  })

  if (aiFallback?.detected && isIntentAligned(aiFallback.text ?? '', detectedIntent, discussionKeywordSet)) {
    return buildPassOutput({
      timestamp: aiFallback.timestamp,
      text: aiFallback.text
    })
  }

  return buildFailOutput()
}
