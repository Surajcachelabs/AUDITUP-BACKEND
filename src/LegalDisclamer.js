import {
  extractTranscriptSegments,
  isCsmSpeaker,
  loadDictionaryPhrases,
  normalizeText
} from './shared.js'
import { runOpenAiIntentFallback } from './openAiFallback.js'

const LEGAL_DISCLAIMER_ANALYSIS_SCOPE = 'full_transcript'

const NOT_ATTORNEY_PATTERNS = [
  /not\s+(?:an\s+|your\s+)?attorneys?/i,
  /i\s+am\s+not\s+(?:an\s+|your\s+)?attorneys?/i,
  /not\s+(?:a\s+|your\s+)?lawyers?/i,
  /i\s+am\s+not\s+(?:a\s+|your\s+)?lawyer/i,
  /our\s+company\s+is\s+not\s+(?:a\s+)?law\s+firm/i,
  /we\s+are\s+not\s+(?:a\s+)?law\s+firm/i,
  /not\s+legal\s+attorneys?/i,
  /not\s+legal\s+representatives?/i,
  /we\s+are\s+not\s+(?:your\s+)?legal\s+representatives?/i,
  /cannot\s+provide\s+legal\s+representation/i
]

const NOT_LEGAL_ADVICE_PATTERNS = [
  /not\s+legal\s+advice/i,
  /does\s+not\s+constitute\s+legal\s+advice/i,
  /should\s+not\s+be\s+(?:taken|treated)\s+as\s+legal\s+(?:advice|counsel)/i,
  /should\s+not\s+be\s+considered\s+legal\s+(?:advice|counsel)/i,
  /anything\s+discussed\s+(?:here|on\s+this\s+call)\s+is\s+not\s+legal\s+advice/i,
  /(?:anything|whatever)\s+(?:i|we)\s+(?:say|discuss)(?:\s+(?:here|today|on\s+this\s+call))?\s+should\s+not\s+be\s+(?:taken|treated|considered)\s+as\s+legal\s+(?:advice|counsel)/i,
  /what(?:ever)?\s+(?:i|we)\s+(?:say|discuss)(?:\s+(?:here|today|on\s+this\s+call))?\s+should\s+not\s+be\s+(?:taken|treated|considered)\s+as\s+legal\s+(?:advice|counsel)/i,
  /general\s+guidance\s+and\s+not\s+legal\s+advice/i,
  /cannot\s+offer\s+legal\s+opinions/i,
  /informational\s+guidance\s*,?\s*not\s+legal\s+advice/i,
  /guidance\s+(?:rather\s+than|not)\s+legal\s+(?:advice|counsel)/i,
  /(?:do\s*not|don't)\s+take\s+(?:this|it|what(?:ever)?\s+we\s+discuss(?:\s+here)?)\s+as\s+legal\s+(?:advice|counsel)/i,
  /(?:do\s*not|don't)\s+consider\s+(?:this|it|that)\s+(?:to\s+be\s+)?legal\s+(?:advice|counsel)/i,
  /(?:this|it|that)\s+(?:is\s+not|isn't)\s+(?:to\s+be\s+)?(?:taken|treated)\s+as\s+legal\s+(?:advice|counsel)/i,
  /not\s+intended\s+as\s+legal\s+(?:advice|counsel)/i,
  /(?:cannot|can\s*not|can't)\s+(?:give|provide|offer)\s+legal\s+(?:advice|counsel)/i,
  /(?:do\s*not|don't)\s+interpret\s+(?:this|it|that)\s+as\s+legal\s+(?:advice|counsel)/i
]

const LEGAL_LIMITATION_CONTEXT_PATTERNS = [
  /you\s+may\s+consult\s+an\s+attorney/i,
  /this\s+call\s+is\s+for\s+informational\s+purposes\s+only/i,
  /nothing\s+discussed\s+(?:here|on\s+this\s+call)\s+is\s+legal\s+advice/i
]

const LEGAL_DISCLAIMER_LLM_INSTRUCTIONS = [
  'Determine whether a Legal Disclaimer is present in these CSM segments.',
  'Evaluate the entire transcript across all CSM segments, not just the ending or closing portion of the call.',
  'Definition: A Legal Disclaimer is present only when the speaker clearly communicates at least one of these intents: (1) they are not attorneys/lawyers/legal representatives, or (2) they cannot provide legal advice/legal counsel.',
  'Mark detected false when the speaker only mentions words like legal, advice, legal advice, attorney, or lawyer in a general tone without clearly disclaiming authority or advice.',
  'Do not infer disclaimer intent from topic mentions alone. Generic legal discussions must not be marked as disclaimer.',
  'Mark detected true if and only if disclaimer intent is explicit or clearly paraphrased.',
  'Return the strongest supporting segment text and timestamp as evidence.'
].join(' ')

const GENERIC_LEGAL_MENTION_PATTERN = /\b(legal|advice|counsel|attorney|attorneys|lawyer|lawyers)\b/i

const LEGAL_TRIGGER_KEYWORD_PATTERN =
  /\b(legal\s+advice|legal\s+counsel|attorney|attorneys|lawyer|lawyers|counsel|legal\s+representative|legal\s+representatives|law\s+firm|advice|legal)\b/i

const PER_SEGMENT_LLM_INSTRUCTIONS = [
  'You are evaluating specific CSM (Customer Success Manager) statements from a client call.',
  'Each statement below was flagged because it contains a legal-related keyword (such as legal advice, advice, attorney, lawyer, counsel, etc.).',
  'For each flagged statement, determine whether the CSM is communicating —in any manner, phrasing, or tone— at least one of these intents:',
  '(1) that they or their company are NOT attorneys, lawyers, or legal representatives, OR',
  '(2) that whatever they discuss should NOT be considered, taken, or treated as legal advice or legal counsel.',
  'The CSM may express this directly, indirectly, formally, casually, or through paraphrase.',
  'Focus on the INTENT behind the words, not merely the presence of keywords.',
  'If the CSM is only discussing a legal topic, referring to an attorney in passing, or using "advice" in a non-disclaimer context (e.g., "my advice is to call back tomorrow"), that is NOT a legal disclaimer.',
  'Mark detected as true ONLY when the intent to disclaim legal authority or disclaim that their words constitute legal advice is clearly present in at least one statement.',
  'If detected is true, return the specific statement that contains the disclaimer intent, its timestamp, and a brief reason explaining why it qualifies as a legal disclaimer.'
].join(' ')

function buildPassOutput({ timestamp, text, reason, decisionSource, llmResult }) {
  return {
    parameter: 'Legal Disclaimer',
    disclaimer_detected: true,
    score: 1,
    status: 'PASS',
    analysis_scope: LEGAL_DISCLAIMER_ANALYSIS_SCOPE,
    disclaimer_timestamp: timestamp,
    disclaimer_text: text,
    reason,
    decision_source: decisionSource,
    ml_detected: Boolean(llmResult?.detected),
    ml_timestamp: llmResult?.timestamp ?? null,
    ml_evidence_text: llmResult?.text ?? null,
    ml_reason: llmResult?.reason ?? null
  }
}

function buildFailOutput({ reason, llmResult } = {}) {
  return {
    parameter: 'Legal Disclaimer',
    disclaimer_detected: false,
    score: 0,
    status: 'FAIL',
    analysis_scope: LEGAL_DISCLAIMER_ANALYSIS_SCOPE,
    reason: reason ?? 'No legal disclaimer provided by the CSM in the transcript.',
    decision_source: llmResult?.detected ? 'llm-negative' : 'rule-negative',
    ml_detected: Boolean(llmResult?.detected),
    ml_timestamp: llmResult?.timestamp ?? null,
    ml_evidence_text: llmResult?.text ?? null,
    ml_reason: llmResult?.reason ?? null
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

function analyzeLegalDisclaimerSignals(text) {
  const attorneySignal = NOT_ATTORNEY_PATTERNS.some((pattern) => pattern.test(text))
  const legalAdviceSignal = NOT_LEGAL_ADVICE_PATTERNS.some((pattern) => pattern.test(text))
  const contextSignal = LEGAL_LIMITATION_CONTEXT_PATTERNS.some((pattern) => pattern.test(text))
  const genericLegalMention = GENERIC_LEGAL_MENTION_PATTERN.test(text)

  const strongIntent = attorneySignal || legalAdviceSignal
  const weakMention = !strongIntent && (contextSignal || genericLegalMention)

  return {
    attorneySignal,
    legalAdviceSignal,
    contextSignal,
    genericLegalMention,
    strongIntent,
    weakMention
  }
}

function hasStrongLegalDisclaimerIntent(text) {
  return analyzeLegalDisclaimerSignals(text).strongIntent
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

  let bestStrictCandidate = null

  for (const segment of csmSegments) {
    const normalizedSegment = normalizeText(segment.text)

    for (const phraseRecord of phraseRecords) {
      const strictMatch =
        normalizedSegment.includes(phraseRecord.normalized) ||
        (phraseRecord.tokenCount === 1 &&
          normalizedSegment.split(' ').includes(phraseRecord.normalized))

      const signalAnalysis = analyzeLegalDisclaimerSignals(segment.text)
      const hasRelevantSignal = signalAnalysis.strongIntent || signalAnalysis.weakMention

      if (!strictMatch || !hasRelevantSignal) {
        continue
      }

      const candidate = {
        score: phraseRecord.tokenCount + (signalAnalysis.strongIntent ? 10 : 0),
        timestamp: segment.timestamp,
        text: segment.text,
        timestampSeconds: segment.timestampSeconds,
        intentStrength: signalAnalysis.strongIntent ? 'strong' : 'weak',
        matchMode: 'repository-strict'
      }

      if (!bestStrictCandidate || candidate.score > bestStrictCandidate.score) {
        bestStrictCandidate = candidate
        continue
      }

      if (
        bestStrictCandidate &&
        candidate.score === bestStrictCandidate.score &&
        segment.timestampSeconds < bestStrictCandidate.timestampSeconds
      ) {
        bestStrictCandidate = candidate
      }
    }
  }

  if (bestStrictCandidate) {
    return {
      matched: true,
      timestamp: bestStrictCandidate.timestamp,
      text: bestStrictCandidate.text,
      intentStrength: bestStrictCandidate.intentStrength,
      matchMode: bestStrictCandidate.matchMode
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
      const signalAnalysis = analyzeLegalDisclaimerSignals(segment.text)
      const hasRelevantSignal = signalAnalysis.strongIntent || signalAnalysis.weakMention
      const weightedScore = score + (signalAnalysis.strongIntent ? 0.1 : 0)

      if (score >= threshold && hasRelevantSignal) {
        if (!bestCandidate || weightedScore > bestCandidate.score) {
          bestCandidate = {
            score: weightedScore,
            timestamp: segment.timestamp,
            text: segment.text,
            timestampSeconds: segment.timestampSeconds,
            intentStrength: signalAnalysis.strongIntent ? 'strong' : 'weak'
          }
        }

        if (
          bestCandidate &&
          weightedScore === bestCandidate.score &&
          segment.timestampSeconds < bestCandidate.timestampSeconds
        ) {
          bestCandidate = {
            score: weightedScore,
            timestamp: segment.timestamp,
            text: segment.text,
            timestampSeconds: segment.timestampSeconds,
            intentStrength: signalAnalysis.strongIntent ? 'strong' : 'weak'
          }
        }
      }
    }
  }

  if (bestCandidate) {
    return {
      matched: true,
      timestamp: bestCandidate.timestamp,
      text: bestCandidate.text,
      intentStrength: bestCandidate.intentStrength,
      matchMode: 'repository-flexible'
    }
  }

  return {
    matched: false,
    timestamp: null,
    text: null,
    intentStrength: 'none',
    matchMode: 'none'
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

  const triggeredSegments = csmSegments.filter((segment) =>
    LEGAL_TRIGGER_KEYWORD_PATTERN.test(segment.text)
  )

  let perSegmentLlmResult = null
  if (triggeredSegments.length > 0) {
    perSegmentLlmResult = await runOpenAiIntentFallback({
      parameterName: 'Legal Disclaimer Per-Sentence Intent Verification',
      instructions: PER_SEGMENT_LLM_INSTRUCTIONS,
      segments: triggeredSegments
    })
  }

  const llmConfirmedIntent = Boolean(perSegmentLlmResult?.detected)

  const transcriptHasStrongIntent = csmSegments.some((segment) =>
    hasStrongLegalDisclaimerIntent(segment.text)
  )

  if (llmConfirmedIntent) {
    const repoConfirmed = repositoryMatch.matched
    let decisionSource = 'per-segment-llm'
    let reason =
      perSegmentLlmResult.reason ||
      'Per-segment ML verification confirmed legal disclaimer intent in a CSM statement.'

    if (repoConfirmed && transcriptHasStrongIntent) {
      decisionSource = 'rule+per-segment-llm'
      reason = `Rule-based patterns and per-segment ML verification both confirmed legal disclaimer intent. ML analysis: ${perSegmentLlmResult.reason}`
    } else if (repoConfirmed) {
      decisionSource = 'repository+per-segment-llm'
      reason = `Repository phrase match found, and per-segment ML verification confirmed disclaimer intent. ML analysis: ${perSegmentLlmResult.reason}`
    }

    return buildPassOutput({
      timestamp: perSegmentLlmResult.timestamp,
      text: perSegmentLlmResult.text,
      reason,
      decisionSource,
      llmResult: perSegmentLlmResult
    })
  }

  if (repositoryMatch.matched && repositoryMatch.intentStrength === 'strong') {
    return buildPassOutput({
      timestamp: repositoryMatch.timestamp,
      text: repositoryMatch.text,
      reason:
        'Rule-based detection found a definitive legal disclaimer pattern in the CSM transcript.',
      decisionSource: 'rule',
      llmResult: perSegmentLlmResult
    })
  }

  if (triggeredSegments.length > 0) {
    return buildFailOutput({
      reason:
        perSegmentLlmResult?.reason ||
        'CSM mentioned legal/advice/attorney terms, but per-segment ML verification did not confirm disclaimer intent in any statement.',
      llmResult: perSegmentLlmResult
    })
  }

  return buildFailOutput({
    reason:
      'No legal disclaimer keywords (legal advice, advice, attorney, lawyer, counsel) were found in any CSM statement.',
    llmResult: perSegmentLlmResult
  })
}
