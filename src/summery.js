import {
  extractTranscriptSegments,
  getTotalCallDurationSeconds,
  isCsmSpeaker,
  loadDictionaryPhrases,
  matchSegmentsAgainstPhrases,
  normalizeText
} from './shared.js'
import { runOpenAiIntentFallback, runOpenAiStructuredSummary } from './openAiFallback.js'

const SUMMARY_ANALYSIS_WINDOW_PERCENTAGE = 35
const SUMMARY_WINDOW_START_RATIO = 1 - SUMMARY_ANALYSIS_WINDOW_PERCENTAGE / 100
const SUMMARY_WORD_REGEX = /\bsummar(?:y|ies|ise|ises|ised|ising|isation|ize|izes|ized|izing|ization)\b/

const SUMMARY_INTENT_CUES = [
  'recap',
  'summary',
  'summarize',
  'summarise',
  'next step',
  'next steps',
  'key point',
  'key points',
  'takeaway',
  'takeaways',
  'action item',
  'action items',
  'overview',
  'wrap up',
  'conclusion',
  'before we close',
  'before we wrap up',
  'highlights'
]

const ISSUE_CUES = [
  'issue',
  'problem',
  'concern',
  'stuck',
  'unable',
  'not able',
  'error',
  'delay',
  'pending',
  'missing',
  'clarify',
  'question',
  'doubt',
  'help'
]

const ACTION_CUES = [
  'i will',
  'we will',
  'let me',
  'i can',
  'we can',
  'review',
  'check',
  'share',
  'send',
  'update',
  'guide',
  'explain',
  'provide',
  'schedule',
  'follow up',
  'create',
  'submit',
  'raise'
]

const RESOLUTION_CUES = [
  'next step',
  'next steps',
  'action item',
  'timeline',
  'eta',
  'plan',
  'proceed',
  'moving forward',
  'follow up',
  'within',
  'by ',
  'reconnect',
  'phase'
]

const OUTCOME_CUES = [
  'agreed',
  'confirmed',
  'resolved',
  'completed',
  'closed',
  'on track',
  'clear direction',
  'understood',
  'finalized',
  'aligned',
  'will reconnect',
  'send a follow up email'
]

const CLOSING_ONLY_CUES = [
  'thank you for your time',
  'thank you',
  'take care',
  'goodbye',
  'have a great day',
  'wishing you',
  'thanks for joining',
  'wonderful evening'
]

const SUMMARY_TRIGGER_KEYWORD_PATTERN =
  /\b(summarize|summarise|summary|summarizing|summarising|recap|recapping|recapped|to\s+summarize|to\s+recap|let\s+me\s+recap|in\s+summary|just\s+to\s+recap|key\s+points?|action\s+items?|next\s+steps?|takeaways?|wrap\s+up|highlights|overview|conclusion|what\s+we\s+discussed|points\s+we\s+covered|go\s+over\s+everything)\b/i

const PER_SEGMENT_SUMMARY_LLM_INSTRUCTIONS = [
  'You are evaluating specific CSM (Customer Success Manager) statements from a client call.',
  'Each statement below was flagged because it contains a keyword related to summarization or recapping (such as summarize, recap, next steps, action items, key points, etc.).',
  'For each flagged statement, determine whether the CSM is expressing the INTENT to recap, summarize, or list the key points discussed during the conversation.',
  'Summarization is considered present when the speaker indicates an intent to recap, summarize, or consolidate previously discussed items.',
  'Examples of intent-based recap statements include:',
  '"Here is what we discussed today."',
  '"These are the points we covered."',
  '"Let me quickly go over everything again."',
  '"These are the action items."',
  '"These are the next steps."',
  'IMPORTANT: Mark detected as true even if the detailed summary points are incomplete or missing, as long as the INTENT to summarize or recap is clearly stated by the CSM.',
  'If the CSM is merely mentioning next steps, action items, or key points as part of regular conversation without the intent to recap or consolidate the discussion, that is NOT summarization.',
  'Focus on the INTENT behind the words, not merely the presence of keywords.',
  'Mark detected as true ONLY when the intent to summarize, recap, or consolidate the conversation is clearly present in at least one statement.',
  'If detected is true, return the specific statement that contains the summarization intent, its timestamp, and a brief reason explaining why it qualifies as summarization.'
].join(' ')

function containsAny(text, cues) {
  return cues.some((cue) => text.includes(cue))
}

function countCueMatches(text, cues) {
  let count = 0

  for (const cue of cues) {
    if (text.includes(cue)) {
      count += 1
    }
  }

  return count
}

function countWords(text) {
  return normalizeText(text).split(' ').filter(Boolean).length
}

function compactText(text, maxLength = 300) {
  const collapsed = String(text ?? '').replace(/\s+/g, ' ').trim()

  if (collapsed.length <= maxLength) {
    return collapsed
  }

  return `${collapsed.slice(0, Math.max(0, maxLength - 3)).trim()}...`
}

function isMeaningfulSegment(text) {
  return countWords(text) >= 4
}

function isClosingOnlySegment(normalizedText) {
  if (!containsAny(normalizedText, CLOSING_ONLY_CUES)) {
    return false
  }

  const hasSubstantiveSignal =
    containsAny(normalizedText, ISSUE_CUES) ||
    containsAny(normalizedText, ACTION_CUES) ||
    containsAny(normalizedText, RESOLUTION_CUES) ||
    containsAny(normalizedText, OUTCOME_CUES)

  return !hasSubstantiveSignal
}

function buildEnrichedSegments(segments) {
  const total = Math.max(1, segments.length - 1)

  return segments.map((segment, index) => {
    const normalizedText = normalizeText(segment.text)

    return {
      ...segment,
      normalizedText,
      isCsm: isCsmSpeaker(segment.speaker),
      relativePosition: index / total
    }
  })
}

function dedupeByNormalizedText(segments) {
  const seen = new Set()
  const deduped = []

  for (const segment of segments) {
    const key = segment.normalizedText
    if (!key || seen.has(key)) {
      continue
    }

    seen.add(key)
    deduped.push(segment)
  }

  return deduped
}

function joinSegmentTexts(segments, maxItems = 2) {
  const selected = dedupeByNormalizedText(segments)
    .slice(0, maxItems)
    .map((segment) => compactText(segment.text, 220))
    .filter(Boolean)

  return selected.join(' | ')
}

function createPhraseRecords(phrases) {
  const uniqueMap = new Map()

  for (const phrase of phrases) {
    const normalized = normalizeText(phrase)
    if (!normalized || uniqueMap.has(normalized)) {
      continue
    }

    uniqueMap.set(normalized, {
      phrase: String(phrase).trim(),
      normalized,
      tokenCount: normalized.split(' ').filter(Boolean).length
    })
  }

  return [...uniqueMap.values()]
}

function hasSummaryWord(text) {
  return SUMMARY_WORD_REGEX.test(text)
}

function hasSummaryIntentCue(text) {
  return containsAny(text, SUMMARY_INTENT_CUES)
}

function evaluateTailSummaryIntent(segments, summaryDictionaryPhrases) {
  const totalDuration = getTotalCallDurationSeconds(segments)
  const summaryWindowStartSeconds = totalDuration * SUMMARY_WINDOW_START_RATIO

  const csmWindowSegments = segments.filter(
    (segment) =>
      isCsmSpeaker(segment.speaker) &&
      typeof segment.timestampSeconds === 'number' &&
      segment.timestampSeconds >= summaryWindowStartSeconds
  )

  if (csmWindowSegments.length === 0) {
    return {
      detected: false,
      timestamp: null,
      evidenceText: null,
      matchedKeywords: [],
      mode: 'none',
      reason: `No CSM segment was found in the final ${SUMMARY_ANALYSIS_WINDOW_PERCENTAGE}% of the call.`
    }
  }

  const phraseRecords = createPhraseRecords(summaryDictionaryPhrases)
  const strictCandidates = []

  for (const segment of csmWindowSegments) {
    const normalizedSegment = normalizeText(segment.text)
    const tokens = normalizedSegment.split(' ').filter(Boolean)
    const tokenSet = new Set(tokens)

    const directMatches = phraseRecords
      .filter(
        (record) =>
          normalizedSegment.includes(record.normalized) ||
          (record.tokenCount === 1 && tokenSet.has(record.normalized))
      )
      .map((record) => record.phrase)

    const summaryWordDetected = hasSummaryWord(normalizedSegment)
    const intentCueDetected = hasSummaryIntentCue(normalizedSegment)

    const hasIntent =
      summaryWordDetected ||
      intentCueDetected ||
      directMatches.some((phrase) => normalizeText(phrase).split(' ').length >= 3)

    if (!hasIntent || (directMatches.length === 0 && !summaryWordDetected)) {
      continue
    }

    strictCandidates.push({
      timestamp: segment.timestamp,
      timestampSeconds: segment.timestampSeconds,
      evidenceText: compactText(segment.text, 260),
      matchedKeywords: directMatches,
      summaryWordDetected,
      intentCueDetected,
      score: directMatches.length + (summaryWordDetected ? 3 : 0) + (intentCueDetected ? 1 : 0)
    })
  }

  if (strictCandidates.length > 0) {
    const bestStrictCandidate = [...strictCandidates].sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score
      }

      return right.timestampSeconds - left.timestampSeconds
    })[0]

    return {
      detected: true,
      timestamp: bestStrictCandidate.timestamp,
      evidenceText: bestStrictCandidate.evidenceText,
      matchedKeywords: bestStrictCandidate.matchedKeywords.slice(0, 8),
      mode: 'rule-strict',
      reason: `CSM summary intent was detected in the final ${SUMMARY_ANALYSIS_WINDOW_PERCENTAGE}% using keyword and phrase evidence.`
    }
  }

  const flexibleMatch = matchSegmentsAgainstPhrases(csmWindowSegments, summaryDictionaryPhrases)

  if (flexibleMatch.matched) {
    const evidenceSegment = csmWindowSegments.find((segment) => segment.timestamp === flexibleMatch.timestamp)
    const normalizedEvidence = normalizeText(evidenceSegment?.text ?? '')
    const summaryWordDetected = hasSummaryWord(normalizedEvidence)
    const intentCueDetected = hasSummaryIntentCue(normalizedEvidence)

    if (summaryWordDetected || intentCueDetected) {
      return {
        detected: true,
        timestamp: flexibleMatch.timestamp,
        evidenceText: evidenceSegment ? compactText(evidenceSegment.text, 260) : null,
        matchedKeywords: flexibleMatch.matchedPhrase ? [flexibleMatch.matchedPhrase] : [],
        mode: 'rule-flexible',
        reason: `CSM summary intent was detected in the final ${SUMMARY_ANALYSIS_WINDOW_PERCENTAGE}% using flexible phrase similarity.`
      }
    }
  }

  return {
    detected: false,
    timestamp: null,
    evidenceText: null,
    matchedKeywords: [],
    mode: 'none',
    reason: `No CSM summarisation intent was detected in the final ${SUMMARY_ANALYSIS_WINDOW_PERCENTAGE}% of the call.`
  }
}

function selectCustomerIssue(enrichedSegments) {
  const direct = enrichedSegments.filter(
    (segment) =>
      !segment.isCsm &&
      isMeaningfulSegment(segment.text) &&
      !isClosingOnlySegment(segment.normalizedText) &&
      (containsAny(segment.normalizedText, ISSUE_CUES) || segment.text.includes('?'))
  )

  if (direct.length > 0) {
    return {
      text: joinSegmentTexts(direct, 2),
      timestamp: direct[0].timestamp,
      sourcedFromTranscript: true
    }
  }

  const fallback = enrichedSegments.filter(
    (segment) => !segment.isCsm && isMeaningfulSegment(segment.text)
  )

  if (fallback.length > 0) {
    return {
      text: joinSegmentTexts(fallback, 1),
      timestamp: fallback[0].timestamp,
      sourcedFromTranscript: true
    }
  }

  return {
    text: 'Client concern was not explicitly articulated in a clear standalone statement.',
    timestamp: null,
    sourcedFromTranscript: false
  }
}

function selectCsmActions(enrichedSegments) {
  const candidates = enrichedSegments
    .filter(
      (segment) =>
        segment.isCsm &&
        isMeaningfulSegment(segment.text) &&
        !isClosingOnlySegment(segment.normalizedText)
    )
    .map((segment) => ({
      ...segment,
      actionScore:
        countCueMatches(segment.normalizedText, ACTION_CUES) +
        (containsAny(segment.normalizedText, RESOLUTION_CUES) ? 1 : 0)
    }))
    .filter((segment) => segment.actionScore > 0)

  if (candidates.length > 0) {
    const ranked = [...candidates].sort((left, right) => {
      if (right.actionScore !== left.actionScore) {
        return right.actionScore - left.actionScore
      }

      return left.timestampSeconds - right.timestampSeconds
    })

    return {
      text: joinSegmentTexts(ranked, 2),
      timestamp: ranked[0].timestamp,
      sourcedFromTranscript: true
    }
  }

  return {
    text: 'CSM actions were discussed but no clear action-oriented statement was detected.',
    timestamp: null,
    sourcedFromTranscript: false
  }
}

function selectResolutionNextSteps(enrichedSegments) {
  const hasTimelineReference = (normalizedText) =>
    /\b\d+\s*(day|days|week|weeks|month|months|hour|hours)\b/.test(normalizedText)

  const candidates = enrichedSegments
    .filter(
      (segment) => isMeaningfulSegment(segment.text) && !isClosingOnlySegment(segment.normalizedText)
    )
    .map((segment) => ({
      ...segment,
      resolutionScore:
        countCueMatches(segment.normalizedText, RESOLUTION_CUES) +
        (hasTimelineReference(segment.normalizedText) ? 1 : 0) +
        (segment.relativePosition >= 0.45 ? 1 : 0)
    }))
    .filter((segment) => segment.resolutionScore > 0)

  if (candidates.length > 0) {
    const ranked = [...candidates].sort((left, right) => {
      if (right.resolutionScore !== left.resolutionScore) {
        return right.resolutionScore - left.resolutionScore
      }

      return right.timestampSeconds - left.timestampSeconds
    })

    return {
      text: joinSegmentTexts(ranked, 2),
      timestamp: ranked[0].timestamp,
      sourcedFromTranscript: true
    }
  }

  return {
    text: 'Resolution path or next steps were not clearly defined in the transcript.',
    timestamp: null,
    sourcedFromTranscript: false
  }
}

function selectFinalOutcome(enrichedSegments) {
  const lateSegments = enrichedSegments.filter((segment) => segment.relativePosition >= 0.7)

  const outcomeCandidates = lateSegments
    .filter(
      (segment) => isMeaningfulSegment(segment.text) && !isClosingOnlySegment(segment.normalizedText)
    )
    .map((segment) => ({
      ...segment,
      outcomeScore:
        countCueMatches(segment.normalizedText, OUTCOME_CUES) +
        countCueMatches(segment.normalizedText, RESOLUTION_CUES) +
        (segment.isCsm ? 1 : 0)
    }))
    .filter((segment) => segment.outcomeScore > 0)

  if (outcomeCandidates.length > 0) {
    const ranked = [...outcomeCandidates].sort((left, right) => {
      if (right.outcomeScore !== left.outcomeScore) {
        return right.outcomeScore - left.outcomeScore
      }

      return right.timestampSeconds - left.timestampSeconds
    })

    return {
      text: joinSegmentTexts(ranked, 1),
      timestamp: ranked[0].timestamp,
      sourcedFromTranscript: true
    }
  }

  const fallback = [...lateSegments].reverse().find(
    (segment) => isMeaningfulSegment(segment.text) && !isClosingOnlySegment(segment.normalizedText)
  )

  if (fallback) {
    return {
      text: compactText(fallback.text, 220),
      timestamp: fallback.timestamp,
      sourcedFromTranscript: true
    }
  }

  return {
    text: 'Final outcome was not explicitly stated in the conversation.',
    timestamp: null,
    sourcedFromTranscript: false
  }
}

function buildHeuristicSummary(segments) {
  const enrichedSegments = buildEnrichedSegments(segments)

  const customerIssue = selectCustomerIssue(enrichedSegments)
  const csmActions = selectCsmActions(enrichedSegments)
  const resolutionNextSteps = selectResolutionNextSteps(enrichedSegments)
  const finalOutcome = selectFinalOutcome(enrichedSegments)

  return {
    customerIssue,
    csmActions,
    resolutionNextSteps,
    finalOutcome
  }
}

function buildOverallSummary({ customerIssue, csmActions, resolutionNextSteps, finalOutcome }) {
  return [
    `Customer issue/problem: ${customerIssue}`,
    `Actions taken by CSM: ${csmActions}`,
    `Resolution/next steps: ${resolutionNextSteps}`,
    `Final outcome: ${finalOutcome}`
  ].join(' ')
}

function pickSummaryTimestamp(sectionTimestamps) {
  return (
    sectionTimestamps.finalOutcome ||
    sectionTimestamps.resolutionNextSteps ||
    sectionTimestamps.csmActions ||
    sectionTimestamps.customerIssue ||
    null
  )
}

function buildOutput({
  customerIssue,
  csmActions,
  resolutionNextSteps,
  finalOutcome,
  sectionTimestamps,
  fieldCoverage,
  summaryIntent,
  decisionSource,
  mlSummary
}) {
  const summaryText = buildOverallSummary({
    customerIssue,
    csmActions,
    resolutionNextSteps,
    finalOutcome
  })

  const summaryDetected = summaryIntent.detected
  const status = summaryDetected ? 'PASS' : 'FAIL'

  const coverageReason =
    fieldCoverage >= 3
      ? `Full-transcript summary evidence captured ${fieldCoverage}/4 required fields.`
      : `Full-transcript summary evidence is partial with ${fieldCoverage}/4 required fields.`

  const matchPreview =
    summaryIntent.matchedKeywords.length > 0
      ? ` Matched cues: ${summaryIntent.matchedKeywords.slice(0, 5).join(', ')}.`
      : ''

  const baseReason = `${summaryIntent.reason}${matchPreview} ${coverageReason}`
  const reason = mlSummary?.reason ? `${baseReason} ML refinement note: ${mlSummary.reason}` : baseReason

  return {
    parameter: 'Call Summarisation',
    summary_detected: summaryDetected,
    score: summaryDetected ? 1 : 0,
    status,
    summary_timestamp: summaryIntent.timestamp,
    summary_text: summaryText,
    reason,
    decision_source: decisionSource,
    analysis_window_percentage: SUMMARY_ANALYSIS_WINDOW_PERCENTAGE,
    summary_intent_match_mode: summaryIntent.mode,
    summary_intent_timestamp: summaryIntent.timestamp,
    summary_intent_evidence_text: summaryIntent.evidenceText,
    summary_intent_matched_keywords: summaryIntent.matchedKeywords,
    full_transcript_summary_timestamp: pickSummaryTimestamp(sectionTimestamps),
    customer_issue: customerIssue,
    csm_actions_taken: csmActions,
    resolution_or_next_steps: resolutionNextSteps,
    final_outcome: finalOutcome,
    summary_field_coverage: fieldCoverage,
    ml_detected: Boolean(mlSummary),
    ml_timestamp: null,
    ml_evidence_text: mlSummary?.overallSummary ?? null,
    ml_reason: mlSummary?.reason ?? null
  }
}

export async function evaluateSummery(transcriptPayload) {
  const segments = extractTranscriptSegments(transcriptPayload)

  if (segments.length === 0) {
    return {
      parameter: 'Call Summarisation',
      summary_detected: false,
      score: 0,
      status: 'FAIL',
      summary_timestamp: null,
      summary_text: null,
      reason: 'No transcript segments were detected, so full-context summarisation could not be generated.',
      decision_source: 'none',
      analysis_window_percentage: SUMMARY_ANALYSIS_WINDOW_PERCENTAGE,
      summary_intent_match_mode: 'none',
      summary_intent_timestamp: null,
      summary_intent_evidence_text: null,
      summary_intent_matched_keywords: [],
      full_transcript_summary_timestamp: null,
      customer_issue: null,
      csm_actions_taken: null,
      resolution_or_next_steps: null,
      final_outcome: null,
      summary_field_coverage: 0,
      ml_detected: false,
      ml_timestamp: null,
      ml_evidence_text: null,
      ml_reason: null
    }
  }

  const csmSegments = segments.filter((segment) => isCsmSpeaker(segment.speaker))

  const heuristicSummary = buildHeuristicSummary(segments)
  const mlSummary = await runOpenAiStructuredSummary({ segments })
  const summaryDictionaryPhrases = await loadDictionaryPhrases([
    'summarization_output.json',
    'summarisation_output.json'
  ]).catch(() => [])

  const ruleIntent = evaluateTailSummaryIntent(segments, summaryDictionaryPhrases)

  const triggeredSegments = csmSegments.filter((segment) =>
    SUMMARY_TRIGGER_KEYWORD_PATTERN.test(segment.text)
  )

  let perSegmentLlmResult = null
  if (triggeredSegments.length > 0) {
    perSegmentLlmResult = await runOpenAiIntentFallback({
      parameterName: 'Summarization Intent Verification',
      instructions: PER_SEGMENT_SUMMARY_LLM_INSTRUCTIONS,
      segments: triggeredSegments
    })
  }

  const llmConfirmedIntent = Boolean(perSegmentLlmResult?.detected)

  const summaryIntent = (() => {
    if (llmConfirmedIntent) {
      const ruleAlsoDetected = ruleIntent.detected
      let mode = 'per-segment-llm'
      let reason =
        perSegmentLlmResult.reason ||
        'Per-segment ML verification confirmed summarization intent in a CSM statement.'

      if (ruleAlsoDetected) {
        mode = 'rule+per-segment-llm'
        reason = `Rule-based patterns and per-segment ML verification both confirmed summarization intent. ML analysis: ${perSegmentLlmResult.reason}`
      }

      return {
        detected: true,
        timestamp: perSegmentLlmResult.timestamp ?? ruleIntent.timestamp,
        evidenceText: perSegmentLlmResult.text ?? ruleIntent.evidenceText,
        matchedKeywords: ruleIntent.matchedKeywords,
        mode,
        reason
      }
    }

    if (ruleIntent.detected) {
      return ruleIntent
    }

    if (triggeredSegments.length > 0) {
      return {
        detected: false,
        timestamp: null,
        evidenceText: null,
        matchedKeywords: [],
        mode: 'none',
        reason:
          perSegmentLlmResult?.reason ||
          'CSM mentioned summarization-related keywords, but per-segment ML verification did not confirm recap/summarization intent in any statement.'
      }
    }

    return {
      detected: false,
      timestamp: null,
      evidenceText: null,
      matchedKeywords: [],
      mode: 'none',
      reason: 'No summarization keywords (summarize, recap, next steps, action items, key points, etc.) were found in any CSM statement.'
    }
  })()

  const customerIssue = mlSummary?.customerIssue ?? heuristicSummary.customerIssue.text
  const csmActions = mlSummary?.csmActions ?? heuristicSummary.csmActions.text
  const resolutionNextSteps =
    mlSummary?.resolutionNextSteps ?? heuristicSummary.resolutionNextSteps.text
  const finalOutcome = mlSummary?.finalOutcome ?? heuristicSummary.finalOutcome.text

  const coverageFlags = {
    customerIssue: Boolean(mlSummary?.customerIssue) || heuristicSummary.customerIssue.sourcedFromTranscript,
    csmActions: Boolean(mlSummary?.csmActions) || heuristicSummary.csmActions.sourcedFromTranscript,
    resolutionNextSteps:
      Boolean(mlSummary?.resolutionNextSteps) ||
      heuristicSummary.resolutionNextSteps.sourcedFromTranscript,
    finalOutcome: Boolean(mlSummary?.finalOutcome) || heuristicSummary.finalOutcome.sourcedFromTranscript
  }

  const fieldCoverage = Object.values(coverageFlags).filter(Boolean).length

  const decisionSource = (() => {
    const parts = []
    if (summaryIntent.mode.includes('llm')) parts.push('per-segment-llm')
    if (summaryIntent.mode.includes('rule')) parts.push('rule')
    if (!parts.length) parts.push('heuristic')
    if (mlSummary) parts.push('structured-llm')
    return parts.join('+') + `:${summaryIntent.mode}`
  })()

  return buildOutput({
    customerIssue,
    csmActions,
    resolutionNextSteps,
    finalOutcome,
    sectionTimestamps: {
      customerIssue: heuristicSummary.customerIssue.timestamp,
      csmActions: heuristicSummary.csmActions.timestamp,
      resolutionNextSteps: heuristicSummary.resolutionNextSteps.timestamp,
      finalOutcome: heuristicSummary.finalOutcome.timestamp
    },
    fieldCoverage,
    summaryIntent,
    decisionSource,
    mlSummary
  })
}
