import { extractTranscriptSegments, isCsmSpeaker, normalizeText } from './shared.js'
import { runOpenAiStructuredSummary } from './openAiFallback.js'

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
  decisionSource,
  mlSummary,
  status
}) {
  const summaryText = buildOverallSummary({
    customerIssue,
    csmActions,
    resolutionNextSteps,
    finalOutcome
  })

  const summaryDetected = status === 'PASS'

  const baseReason =
    summaryDetected
      ? `Full-transcript intent summary generated with ${fieldCoverage}/4 required fields captured.`
      : `Summary is partial. Only ${fieldCoverage}/4 required fields were confidently captured from transcript context.`

  const reason = mlSummary?.reason ? `${baseReason} ML refinement note: ${mlSummary.reason}` : baseReason

  return {
    parameter: 'Call Summarisation',
    summary_detected: summaryDetected,
    score: summaryDetected ? 1 : 0,
    status,
    summary_timestamp: pickSummaryTimestamp(sectionTimestamps),
    summary_text: summaryText,
    reason,
    decision_source: decisionSource,
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

  const heuristicSummary = buildHeuristicSummary(segments)
  const mlSummary = await runOpenAiStructuredSummary({ segments })

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
  const status = fieldCoverage >= 3 ? 'PASS' : 'FAIL'

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
    decisionSource: mlSummary ? 'heuristic+llm' : 'heuristic',
    mlSummary,
    status
  })
}
