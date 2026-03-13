import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { extractTranscriptSegments, isCsmSpeaker, normalizeText } from './shared.js'
import { runOpenAiEmpathyStressDetection } from './openAiFallback.js'

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url))
const EMPATHY_DATA_PATH = path.join(MODULE_DIR, '..', 'Json', 'empathy_data.json')

const MAX_EMPATHY_EVIDENCE_SECTIONS = 4
const MAX_RESPONSE_SEGMENTS_PER_TRIGGER = 3

const STOP_WORDS = new Set([
  'the',
  'and',
  'for',
  'with',
  'that',
  'this',
  'your',
  'you',
  'are',
  'was',
  'were',
  'have',
  'has',
  'had',
  'will',
  'would',
  'should',
  'could',
  'there',
  'here',
  'about',
  'case',
  'regarding',
  'stage',
  'situation',
  'please',
  'note',
  'let',
  'clarify',
  'just',
  'okay',
  'our',
  'their',
  'them',
  'from',
  'into',
  'what',
  'when'
])

const EMOTIONAL_TRIGGER_PATTERNS = [
  'frustrated',
  'frustrating',
  'stress',
  'stressed',
  'concern',
  'concerned',
  'confused',
  'confusing',
  'worry',
  'worried',
  'disappointed',
  'urgent',
  'urgency',
  'anxious',
  'anxiety',
  'overwhelmed',
  'afraid',
  'fear'
]

const DIFFICULTY_TRIGGER_PATTERNS = [
  'cannot upload',
  'cant upload',
  'unable to upload',
  'cannot submit',
  'cant submit',
  'cannot understand',
  'dont understand',
  'do not understand',
  'delay',
  'taking so long',
  'timeline',
  'deadline',
  'technical issue',
  'technical problem',
  'error',
  'financial concern',
  'fear of rejection',
  'fear of denial',
  'unable to complete',
  'cannot complete',
  'cant complete'
]

const TRIGGER_REGEX_RULES = [
  /\b(cannot|cant|unable|not able)\b.*\b(upload|submit|understand|complete|finish|proceed|document|docs|steps?)\b/,
  /\b(why|how)\b.*\b(taking so long|delay|stuck|not moving)\b/,
  /\b(really|very|so)\b.*\b(worried|stressed|confused|concerned|frustrated)\b/
]

const BASIC_ACKNOWLEDGEMENT_PATTERNS = [
  'i understand',
  'got it',
  'thanks for sharing',
  'i see',
  'i see what you re saying',
  'noted',
  'okay i understand',
  'understood',
  'i get your point'
]

const EMOTIONAL_RECOGNITION_PATTERNS = [
  'this delay is frustrating',
  'that must be concerning',
  'that sounds stressful',
  'i can see why that would be stressful',
  'i can imagine this feels overwhelming',
  'i understand how important this is',
  'i see why this would cause concern',
  'i understand this situation is difficult',
  'i understand this can be frustrating',
  'i completely understand your worry',
  'i hear your frustration'
]

const STRONG_SUPPORT_PATTERNS = [
  'we will work together',
  'we are actively working on this',
  'we are prioritizing your case',
  'i will ensure',
  'i ll ensure',
  'i will personally ensure',
  'i ll personally ensure',
  'i will stay with you',
  'we will guide you',
  'i am here to support',
  'i m here to support',
  'i am here to help',
  'i m here to help',
  'we will resolve this',
  'we ll resolve this',
  'we will keep you updated',
  'we ll keep you updated',
  'we will do everything'
]

const DISMISSIVE_OR_PROCEDURAL_PATTERNS = [
  'that is company policy',
  'that is not our responsibility',
  'there is nothing we can do',
  'you need to submit the documents',
  'you will have to wait',
  'just upload the files',
  'please follow the checklist',
  'that is the process',
  'this is how it works',
  'cannot help',
  'not possible'
]

const CONTRAST_CONNECTOR_PATTERNS = ['but', 'however', 'unfortunately']

let cachedDataset = null

function parseEmpathyEntryScore(entry) {
  const rawScore = entry?.['Empathy Level/Score'] ?? entry?.empathyScore ?? entry?.score
  const numericScore = Number(rawScore)

  if (!Number.isInteger(numericScore) || numericScore < 0 || numericScore > 3) {
    return null
  }

  return numericScore
}

function extractTokens(value) {
  return normalizeText(value)
    .split(' ')
    .filter((token) => token.length >= 3 && !STOP_WORDS.has(token))
}

function collectPatternMatches(normalizedText, patterns) {
  return patterns.filter((pattern) => normalizedText.includes(pattern))
}

function hasAnyPattern(normalizedText, patterns) {
  return patterns.some((pattern) => normalizedText.includes(pattern))
}

function computeTokenSetSimilarity(responseTokenSet, candidateTokenSet) {
  if (responseTokenSet.size === 0 || candidateTokenSet.size === 0) {
    return 0
  }

  let intersectionCount = 0
  for (const token of responseTokenSet) {
    if (candidateTokenSet.has(token)) {
      intersectionCount += 1
    }
  }

  if (intersectionCount === 0) {
    return 0
  }

  const unionCount = responseTokenSet.size + candidateTokenSet.size - intersectionCount
  return intersectionCount / unionCount
}

function buildDatasetIndex(entries) {
  const entriesByScore = {
    0: [],
    1: [],
    2: [],
    3: []
  }
  const textSetByScore = {
    0: new Set(),
    1: new Set(),
    2: new Set(),
    3: new Set()
  }

  for (const entry of entries) {
    const score = parseEmpathyEntryScore(entry)
    const text = entry?.Data

    if (score == null || typeof text !== 'string') {
      continue
    }

    const normalizedText = normalizeText(text)
    if (!normalizedText || textSetByScore[score].has(normalizedText)) {
      continue
    }

    textSetByScore[score].add(normalizedText)

    entriesByScore[score].push({
      text: text.trim(),
      normalizedText,
      tokenSet: new Set(extractTokens(normalizedText))
    })
  }

  return {
    entriesByScore,
    textSetByScore,
    totalEntries:
      entriesByScore[0].length +
      entriesByScore[1].length +
      entriesByScore[2].length +
      entriesByScore[3].length
  }
}

async function loadEmpathyDataset() {
  if (cachedDataset) {
    return cachedDataset
  }

  const rawText = await fs.readFile(EMPATHY_DATA_PATH, 'utf8')
  const parsed = JSON.parse(rawText)

  if (!Array.isArray(parsed)) {
    throw new Error('Invalid empathy_data.json format. Expected array.')
  }

  cachedDataset = buildDatasetIndex(parsed)
  return cachedDataset
}

function buildLlmStressIndex(segments, llmStressSegments) {
  const stressByIndex = new Map()

  if (!Array.isArray(llmStressSegments)) {
    return stressByIndex
  }

  for (const item of llmStressSegments) {
    const index = Number(item?.index)

    if (!Number.isInteger(index) || index < 0 || index >= segments.length) {
      continue
    }

    if (isCsmSpeaker(segments[index]?.speaker)) {
      continue
    }

    if (!stressByIndex.has(index)) {
      stressByIndex.set(index, {
        emotion: typeof item?.emotion === 'string' ? item.emotion.trim() : '',
        reason: typeof item?.reason === 'string' ? item.reason.trim() : ''
      })
    }
  }

  return stressByIndex
}

function detectTriggerSignals(clientText) {
  const normalizedClientText = normalizeText(clientText)
  const emotionalMatches = collectPatternMatches(normalizedClientText, EMOTIONAL_TRIGGER_PATTERNS)
  const difficultyMatches = collectPatternMatches(normalizedClientText, DIFFICULTY_TRIGGER_PATTERNS)
  const regexMatchCount = TRIGGER_REGEX_RULES.filter((rule) => rule.test(normalizedClientText)).length

  const triggerSignals = []
  if (emotionalMatches.length > 0) {
    triggerSignals.push('emotional_signal')
  }
  if (difficultyMatches.length > 0 || regexMatchCount > 0) {
    triggerSignals.push('difficulty_signal')
  }

  return {
    hasTrigger: triggerSignals.length > 0,
    triggerSignals,
    emotionalMatches,
    difficultyMatches,
    regexMatchCount
  }
}

function findBestDatasetSimilarity(normalizedResponse, responseTokenSet, dataset) {
  const bestByScore = {
    0: { similarity: 0, text: null },
    1: { similarity: 0, text: null },
    2: { similarity: 0, text: null },
    3: { similarity: 0, text: null }
  }

  for (let score = 0; score <= 3; score += 1) {
    if (dataset.textSetByScore[score].has(normalizedResponse)) {
      bestByScore[score] = {
        similarity: 1,
        text: normalizedResponse
      }
      continue
    }

    for (const candidate of dataset.entriesByScore[score]) {
      const similarity = computeTokenSetSimilarity(responseTokenSet, candidate.tokenSet)
      if (similarity > bestByScore[score].similarity) {
        bestByScore[score] = {
          similarity,
          text: candidate.text
        }
      }
    }
  }

  let bestScore = 0
  let bestSimilarity = 0
  let bestText = null

  for (let score = 0; score <= 3; score += 1) {
    const candidate = bestByScore[score]
    if (
      candidate.similarity > bestSimilarity ||
      (candidate.similarity === bestSimilarity && score > bestScore)
    ) {
      bestScore = score
      bestSimilarity = candidate.similarity
      bestText = candidate.text
    }
  }

  return {
    bestScore,
    bestSimilarity,
    bestText,
    bestByScore
  }
}

function collectTriggerResponseInteractions(segments) {
  const interactions = []
  let pendingTriggerSegments = []

  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index]

    if (isCsmSpeaker(segment.speaker)) {
      if (pendingTriggerSegments.length === 0) {
        continue
      }

      const responseSegments = []
      let responseIndex = index

      while (responseIndex < segments.length && isCsmSpeaker(segments[responseIndex].speaker)) {
        if (responseSegments.length < MAX_RESPONSE_SEGMENTS_PER_TRIGGER) {
          responseSegments.push(segments[responseIndex])
        }

        responseIndex += 1
      }

      interactions.push({
        triggerSegments: pendingTriggerSegments,
        responseSegments
      })

      pendingTriggerSegments = []
      index = responseIndex - 1
      continue
    }

    const triggerEvaluation = detectTriggerSignals(segment.text)

    if (!triggerEvaluation.hasTrigger) {
      continue
    }

    pendingTriggerSegments.push({
      ...segment,
      trigger_signals: triggerEvaluation.triggerSignals,
      matched_trigger_phrases: [
        ...triggerEvaluation.emotionalMatches,
        ...triggerEvaluation.difficultyMatches
      ],
      stress_detection_source: 'rule',
      stress_reason: '',
      stress_emotion: ''
    })
  }

  return interactions
}

function collectLlmPreprocessedInteractions(segments, llmStressSegments) {
  const stressByIndex = buildLlmStressIndex(segments, llmStressSegments)

  if (stressByIndex.size === 0) {
    return []
  }

  const interactions = []
  let pendingTriggerSegments = []

  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index]

    if (isCsmSpeaker(segment.speaker)) {
      if (pendingTriggerSegments.length === 0) {
        continue
      }

      const responseSegments = []
      let responseIndex = index

      while (responseIndex < segments.length && isCsmSpeaker(segments[responseIndex].speaker)) {
        if (responseSegments.length < MAX_RESPONSE_SEGMENTS_PER_TRIGGER) {
          responseSegments.push(segments[responseIndex])
        }

        responseIndex += 1
      }

      interactions.push({
        triggerSegments: pendingTriggerSegments,
        responseSegments,
        stressDetectionSource: 'llm'
      })

      pendingTriggerSegments = []
      index = responseIndex - 1
      continue
    }

    if (!stressByIndex.has(index)) {
      continue
    }

    const llmStress = stressByIndex.get(index)
    const triggerEvaluation = detectTriggerSignals(segment.text)

    pendingTriggerSegments.push({
      ...segment,
      trigger_signals: [...new Set(['llm_stress_detected', ...triggerEvaluation.triggerSignals])],
      matched_trigger_phrases: [
        ...new Set([
          ...triggerEvaluation.emotionalMatches,
          ...triggerEvaluation.difficultyMatches,
          llmStress.emotion || null
        ].filter(Boolean))
      ],
      stress_detection_source: 'llm',
      stress_reason: llmStress.reason,
      stress_emotion: llmStress.emotion
    })
  }

  return interactions
}

async function collectPreprocessedTriggerResponseInteractions(segments) {
  const llmStressSegments = await runOpenAiEmpathyStressDetection({ segments })

  if (Array.isArray(llmStressSegments)) {
    return {
      source: 'llm',
      llmStressCount: llmStressSegments.length,
      interactions: collectLlmPreprocessedInteractions(segments, llmStressSegments)
    }
  }

  return {
    source: 'rule_fallback',
    llmStressCount: 0,
    interactions: collectTriggerResponseInteractions(segments)
  }
}

function evaluateSingleResponse(interaction, dataset) {
  const responseText = interaction.responseSegments.map((segment) => segment.text).join(' ').trim()
  const normalizedResponse = normalizeText(responseText)
  const responseTokenSet = new Set(extractTokens(normalizedResponse))

  const basicMatches = collectPatternMatches(normalizedResponse, BASIC_ACKNOWLEDGEMENT_PATTERNS)
  const emotionalMatches = collectPatternMatches(normalizedResponse, EMOTIONAL_RECOGNITION_PATTERNS)
  const supportMatches = collectPatternMatches(normalizedResponse, STRONG_SUPPORT_PATTERNS)
  const dismissiveMatches = collectPatternMatches(normalizedResponse, DISMISSIVE_OR_PROCEDURAL_PATTERNS)
  const hasContrastConnector = hasAnyPattern(normalizedResponse, CONTRAST_CONNECTOR_PATTERNS)

  const datasetSimilarity = findBestDatasetSimilarity(normalizedResponse, responseTokenSet, dataset)
  const datasetSuggestedScore =
    datasetSimilarity.bestSimilarity >= 0.58 ? datasetSimilarity.bestScore : null

  const hasDismissiveTone =
    dismissiveMatches.length > 0 &&
    emotionalMatches.length === 0 &&
    supportMatches.length === 0 &&
    (basicMatches.length === 0 || hasContrastConnector)

  let score = 0
  let classLabel = 'no_empathy'

  if (hasDismissiveTone) {
    score = 0
    classLabel = 'dismissive_or_procedural'
  } else if (emotionalMatches.length > 0 && supportMatches.length > 0) {
    score = 3
    classLabel = 'strong_empathy_with_support'
  } else if (emotionalMatches.length > 0) {
    score = 2
    classLabel = 'emotional_recognition'
  } else if (basicMatches.length > 0) {
    score = 1
    classLabel = 'basic_acknowledgement'
  }

  if (
    score <= 1 &&
    datasetSuggestedScore != null &&
    datasetSuggestedScore >= 2 &&
    dismissiveMatches.length === 0 &&
    datasetSimilarity.bestSimilarity >= 0.66
  ) {
    score = datasetSuggestedScore === 3 && supportMatches.length > 0 ? 3 : 2
    classLabel = score === 3 ? 'strong_empathy_with_support' : 'emotional_recognition'
  }

  if (
    dismissiveMatches.length > 0 &&
    hasContrastConnector &&
    supportMatches.length === 0 &&
    emotionalMatches.length <= 1
  ) {
    score = 0
    classLabel = 'dismissive_or_procedural'
  }

  const matchedSignals = []
  if (basicMatches.length > 0) {
    matchedSignals.push('basic_acknowledgement')
  }
  if (emotionalMatches.length > 0) {
    matchedSignals.push('emotional_recognition')
  }
  if (supportMatches.length > 0) {
    matchedSignals.push('support_commitment')
  }
  if (dismissiveMatches.length > 0) {
    matchedSignals.push('dismissive_or_procedural')
  }
  if (datasetSuggestedScore != null) {
    matchedSignals.push('dataset_similarity')
  }

  const matchedPhrases = [
    ...new Set(
      [
        ...basicMatches,
        ...emotionalMatches,
        ...supportMatches,
        ...dismissiveMatches,
        datasetSimilarity.bestText ? `dataset_match: ${datasetSimilarity.bestText}` : null
      ].filter(Boolean)
    )
  ]

  return {
    score,
    classLabel,
    responseText,
    matchedSignals,
    matchedPhrases,
    datasetSuggestedScore,
    datasetSimilarity: datasetSimilarity.bestSimilarity,
    datasetMatchedText: datasetSimilarity.bestText,
    stressDetectionSource: interaction.stressDetectionSource ?? interaction.triggerSegments[0]?.stress_detection_source ?? 'rule',
    triggerSegments: interaction.triggerSegments,
    responseSegments: interaction.responseSegments
  }
}

function summarizeScores(scoredResponses) {
  const counters = {
    0: 0,
    1: 0,
    2: 0,
    3: 0
  }

  let totalScore = 0
  let dismissiveCount = 0
  let strongCount = 0
  let basicOnlyCount = 0

  for (const response of scoredResponses) {
    counters[response.score] += 1
    totalScore += response.score

    if (response.classLabel === 'dismissive_or_procedural' || response.classLabel === 'no_empathy') {
      dismissiveCount += 1
    }
    if (response.classLabel === 'strong_empathy_with_support') {
      strongCount += 1
    }
    if (response.classLabel === 'basic_acknowledgement') {
      basicOnlyCount += 1
    }
  }

  const averageScore = scoredResponses.length > 0 ? totalScore / scoredResponses.length : 0
  let finalBand = Math.round(averageScore)

  if (strongCount === scoredResponses.length && scoredResponses.length > 0) {
    finalBand = 3
  }

  if (dismissiveCount > 0 && finalBand === 3) {
    finalBand = 2
  }

  if (dismissiveCount >= Math.ceil(scoredResponses.length / 2) && finalBand > 1) {
    finalBand = 1
  }

  if (basicOnlyCount === scoredResponses.length && finalBand > 1) {
    finalBand = 1
  }

  finalBand = Math.max(0, Math.min(3, finalBand))

  return {
    finalBand,
    averageScore,
    counters,
    dismissiveCount,
    strongCount,
    basicOnlyCount
  }
}

function buildReason({
  triggerCount,
  evaluatedResponseCount,
  summary,
  dataset,
  topResponse,
  missedTriggerCount,
  preprocessingSource,
  llmStressCount
}) {
  const parts = [
    `Detected ${triggerCount} empathy trigger(s) from client messages and evaluated ${evaluatedResponseCount} immediate CSM response block(s).`,
    `Distribution -> score_0:${summary.counters[0]}, score_1:${summary.counters[1]}, score_2:${summary.counters[2]}, score_3:${summary.counters[3]}, average:${summary.averageScore.toFixed(2)}.`,
    `Empathy dataset from Json/empathy_data.json was used for semantic similarity support (${dataset.totalEntries} unique examples).`
  ]

  if (preprocessingSource === 'llm') {
    parts.unshift(`LLM preprocessing identified ${llmStressCount} client stress segment(s) before empathy scoring.`)
  } else {
    parts.unshift('LLM preprocessing was unavailable, so the rule-based trigger detector was used as fallback before empathy scoring.')
  }

  if (missedTriggerCount > 0) {
    parts.push(`${missedTriggerCount} trigger(s) had no immediate CSM response to evaluate.`)
  }

  if (topResponse && topResponse.datasetMatchedText) {
    parts.push(
      `Closest dataset example: "${topResponse.datasetMatchedText}" (similarity ${topResponse.datasetSimilarity.toFixed(2)}).`
    )
  }

  parts.push(`Final empathy band selected: ${summary.finalBand}.`)

  return parts.join(' ')
}

function buildNaOutput(reason) {
  return {
    parameter: 'Empathy',
    empathy_score: null,
    score: 0,
    status: 'N/A',
    trigger_detected: false,
    trigger_count: 0,
    evaluated_responses: 0,
    category_counters: {
      category_0: 0,
      category_1: 0,
      category_2: 0,
      category_3: 0
    },
    empathy_validation: {
      passed: false,
      framework_version: 'trigger-response-v2',
      scoring_mode: 'not_applicable'
    },
    evidence_sections: [],
    reason
  }
}

export async function evaluateEmpathy(transcriptPayload) {
  const segments = extractTranscriptSegments(transcriptPayload)

  if (!Array.isArray(segments) || segments.length === 0) {
    return buildNaOutput('No transcript segments found for empathy evaluation.')
  }

  let dataset
  try {
    dataset = await loadEmpathyDataset()
  } catch {
    return buildNaOutput('Empathy dataset could not be loaded from Json/empathy_data.json.')
  }

  const preprocessing = await collectPreprocessedTriggerResponseInteractions(segments)
  const triggerResponseInteractions = preprocessing.interactions
  const triggerCount = triggerResponseInteractions.reduce(
    (count, item) => count + item.triggerSegments.length,
    0
  )

  if (triggerCount === 0) {
    return buildNaOutput(
      preprocessing.source === 'llm'
        ? 'LLM preprocessing did not detect any client stress statement requiring empathy evaluation.'
        : 'No empathy trigger was detected in client messages (concern, frustration, confusion, urgency, or completion difficulty).'
    )
  }

  const scoredResponses = triggerResponseInteractions.map((interaction) =>
    evaluateSingleResponse(interaction, dataset)
  )

  if (scoredResponses.length === 0) {
    return buildNaOutput(
      'Empathy trigger(s) were detected, but no immediate CSM response block was found for scoring.'
    )
  }

  const summary = summarizeScores(scoredResponses)
  const sortedByConfidence = [...scoredResponses].sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score
    }
    return right.datasetSimilarity - left.datasetSimilarity
  })
  const topResponse = sortedByConfidence[0] ?? null

  const evidenceSections = sortedByConfidence.slice(0, MAX_EMPATHY_EVIDENCE_SECTIONS).map((item) => {
    const firstResponseSegment = item.responseSegments[0]
    const triggerText = item.triggerSegments.map((segment) => segment.text).join(' | ')

    return {
      timestamp: firstResponseSegment?.timestamp ?? null,
      speaker: firstResponseSegment?.speaker ?? null,
      text: item.responseText,
      client_stress_statement: triggerText,
      matched_signals: item.matchedSignals,
      matched_phrases: item.matchedPhrases,
      trigger_text: triggerText,
      trigger_signals: [...new Set(item.triggerSegments.flatMap((segment) => segment.trigger_signals))],
      stress_detection_source: item.stressDetectionSource,
      stress_reasons: item.triggerSegments.map((segment) => segment.stress_reason).filter(Boolean),
      response_score: item.score,
      response_class: item.classLabel,
      dataset_similarity: Number(item.datasetSimilarity.toFixed(4))
    }
  })

  return {
    parameter: 'Empathy',
    empathy_score: summary.finalBand,
    score: summary.finalBand,
    status: `BAND ${summary.finalBand}`,
    trigger_detected: true,
    trigger_count: triggerCount,
    evaluated_responses: scoredResponses.length,
    category_counters: {
      category_0: summary.counters[0],
      category_1: summary.counters[1],
      category_2: summary.counters[2],
      category_3: summary.counters[3]
    },
    empathy_validation: {
      passed: summary.finalBand > 0,
      framework_version: 'trigger-response-v3-llm-preprocessed',
      scoring_mode: 'rule_plus_dataset_similarity',
      preprocessing_source: preprocessing.source,
      llm_detected_stress_segments: preprocessing.llmStressCount,
      average_response_score: Number(summary.averageScore.toFixed(2)),
      dismissive_response_count: summary.dismissiveCount,
      strong_support_response_count: summary.strongCount,
      basic_acknowledgement_response_count: summary.basicOnlyCount
    },
    evidence_sections: evidenceSections,
    reason: buildReason({
      triggerCount,
      evaluatedResponseCount: scoredResponses.length,
      summary,
      dataset,
      topResponse,
      missedTriggerCount: triggerCount - scoredResponses.length,
      preprocessingSource: preprocessing.source,
      llmStressCount: preprocessing.llmStressCount
    })
  }
}
