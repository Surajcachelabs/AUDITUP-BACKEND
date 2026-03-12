import fs from 'node:fs/promises'
import path from 'node:path'
import { extractTranscriptSegments, isCsmSpeaker, normalizeText } from './shared.js'

const EMPATHY_DATA_PATH = path.join(process.cwd(), 'Json', 'empathy_data.json')

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
  'okay'
])

const POSITIVE_PATTERNS = [
  'i understand',
  'i understand your concern',
  'i hear your frustration',
  'i can see why',
  'i can imagine',
  'that sounds stressful',
  'that must be concerning',
  'sorry',
  'support you',
  'work together',
  'we are actively working',
  'your concern is valid',
  'we are prioritizing',
  'guiding you',
  'thanks for sharing'
]

const NEGATIVE_PATTERNS = [
  'that is company policy',
  'that is not our responsibility',
  'there is nothing we can do',
  'you need to submit',
  'just upload',
  'you will have to wait',
  'please follow the checklist',
  'cannot',
  'cant',
  'not possible'
]

const UNDERSTANDING_PATTERNS = [
  'i understand',
  'i completely understand',
  'i can understand',
  'i get that',
  'i get it',
  'i hear you',
  'i hear your',
  'i can see why',
  'i can imagine'
]

const ACKNOWLEDGEMENT_PATTERNS = [
  'your concern',
  'your frustration',
  'your inconvenience',
  'your situation',
  'i get that',
  'i know this is',
  'i know it can be',
  'this is frustrating',
  'that sounds stressful',
  'that must be concerning',
  'that must be difficult',
  'i hear your concern',
  'i hear your frustration'
]

const EMPATHETIC_PHRASE_PATTERNS = [
  'i understand',
  'i am sorry',
  'i m sorry',
  'im sorry',
  'sorry for the inconvenience',
  'sorry about that',
  'i apologize',
  'i appreciate your patience',
  'thank you for sharing',
  'your concern is valid',
  'we are here to help',
  'i m here to help',
  'i am here to help'
]

const SUPPORT_ACTION_PATTERNS = [
  'i will check',
  'i ll check',
  'i will get that checked',
  'i ll get that checked',
  'i will do that',
  'i ll do that',
  'i will assist',
  'i ll assist',
  'i can assist',
  'i will remind you',
  'i ll remind you',
  'i will connect you',
  'i ll connect you',
  'i will help',
  'i ll help',
  'let me check',
  'let me see what i can do',
  'i will share my screen',
  'i ll share my screen'
]

const MAX_EMPATHY_EVIDENCE_SECTIONS = 4

let cachedLexicon = null

function parseEmpathyEntryScore(entry) {
  const rawScore = entry?.['Empathy Level/Score'] ?? entry?.empathyScore ?? entry?.score
  const numeric = Number(rawScore)

  if (!Number.isInteger(numeric) || numeric < 0 || numeric > 3) {
    return null
  }

  return numeric
}

function extractTokens(value) {
  return normalizeText(value)
    .split(' ')
    .filter((token) => token.length >= 3 && !STOP_WORDS.has(token))
}

function buildLexicon(entries) {
  const levelKeywordSets = {
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

    const tokens = extractTokens(text)

    for (const token of tokens) {
      levelKeywordSets[score].add(token)
    }
  }

  return levelKeywordSets
}

async function loadEmpathyLexicon() {
  if (cachedLexicon) {
    return cachedLexicon
  }

  const rawText = await fs.readFile(EMPATHY_DATA_PATH, 'utf8')
  const parsed = JSON.parse(rawText)

  if (!Array.isArray(parsed)) {
    throw new Error('Invalid empathy_data.json format. Expected array.')
  }

  cachedLexicon = buildLexicon(parsed)
  return cachedLexicon
}

function analyzeSentiment(segments) {
  let positiveHits = 0
  let negativeHits = 0

  for (const segment of segments) {
    const normalized = normalizeText(segment.text)

    for (const phrase of POSITIVE_PATTERNS) {
      if (normalized.includes(phrase)) {
        positiveHits += 1
      }
    }

    for (const phrase of NEGATIVE_PATTERNS) {
      if (normalized.includes(phrase)) {
        negativeHits += 1
      }
    }
  }

  const sentimentScore = positiveHits - negativeHits

  let band = 1
  if (sentimentScore <= -2) {
    band = 0
  } else if (sentimentScore <= 0) {
    band = 1
  } else if (sentimentScore <= 3) {
    band = 2
  } else {
    band = 3
  }

  return {
    sentimentScore,
    band,
    positiveHits,
    negativeHits
  }
}

function countEmpathyCategoryHits(segments, levelKeywordSets) {
  const counters = {
    0: 0,
    1: 0,
    2: 0,
    3: 0
  }

  const matchedKeywords = {
    0: new Set(),
    1: new Set(),
    2: new Set(),
    3: new Set()
  }

  for (const segment of segments) {
    const tokens = normalizeText(segment.text).split(' ').filter(Boolean)

    for (const token of tokens) {
      for (let level = 0; level <= 3; level += 1) {
        if (levelKeywordSets[level].has(token)) {
          counters[level] += 1
          matchedKeywords[level].add(token)
        }
      }
    }
  }

  return {
    counters,
    matchedKeywords
  }
}

function selectDominantBand(counters) {
  let selectedBand = 0
  let maxCount = -1

  for (let level = 0; level <= 3; level += 1) {
    const count = counters[level]

    if (count > maxCount || (count === maxCount && level > selectedBand)) {
      selectedBand = level
      maxCount = count
    }
  }

  return {
    selectedBand,
    maxCount
  }
}

function collectMatchedPhrases(segments, patterns) {
  const matches = new Set()

  for (const segment of segments) {
    const normalized = normalizeText(segment.text)
    for (const phrase of patterns) {
      if (normalized.includes(phrase)) {
        matches.add(phrase)
      }
    }
  }

  return matches
}

function segmentHasAnyPattern(normalizedSegment, patterns) {
  return patterns.some((phrase) => normalizedSegment.includes(phrase))
}

function collectPatternMatches(normalizedSegment, patterns) {
  return patterns.filter((phrase) => normalizedSegment.includes(phrase))
}

function collectEmpathyEvidenceSections(segments, levelKeywordSets) {
  const scoredSections = []

  for (const segment of segments) {
    const normalized = normalizeText(segment.text)
    const tokens = normalized.split(' ').filter(Boolean)

    const understandingMatches = collectPatternMatches(normalized, UNDERSTANDING_PATTERNS)
    const acknowledgementMatches = collectPatternMatches(normalized, ACKNOWLEDGEMENT_PATTERNS)
    const empatheticPhraseMatches = collectPatternMatches(normalized, EMPATHETIC_PHRASE_PATTERNS)
    const supportActionMatches = collectPatternMatches(normalized, SUPPORT_ACTION_PATTERNS)

    const lexiconHits = {
      category_0: 0,
      category_1: 0,
      category_2: 0,
      category_3: 0
    }

    for (const token of tokens) {
      if (levelKeywordSets[0].has(token)) {
        lexiconHits.category_0 += 1
      }
      if (levelKeywordSets[1].has(token)) {
        lexiconHits.category_1 += 1
      }
      if (levelKeywordSets[2].has(token)) {
        lexiconHits.category_2 += 1
      }
      if (levelKeywordSets[3].has(token)) {
        lexiconHits.category_3 += 1
      }
    }

    const lexiconTotalHits =
      lexiconHits.category_0 + lexiconHits.category_1 + lexiconHits.category_2 + lexiconHits.category_3

    const matchedSignals = []
    if (understandingMatches.length > 0) {
      matchedSignals.push('understanding')
    }
    if (acknowledgementMatches.length > 0) {
      matchedSignals.push('acknowledgement')
    }
    if (empatheticPhraseMatches.length > 0) {
      matchedSignals.push('empathetic_phrase')
    }
    if (supportActionMatches.length > 0) {
      matchedSignals.push('support_action')
    }

    if (matchedSignals.length === 0 && lexiconTotalHits === 0) {
      continue
    }

    const evidenceScore =
      understandingMatches.length * 3 +
      acknowledgementMatches.length * 3 +
      empatheticPhraseMatches.length * 4 +
      supportActionMatches.length * 2 +
      lexiconTotalHits

    const matchedPhrases = [
      ...new Set([
        ...understandingMatches,
        ...acknowledgementMatches,
        ...empatheticPhraseMatches,
        ...supportActionMatches
      ])
    ]

    scoredSections.push({
      timestampSeconds: Number(segment.timestampSeconds ?? 0),
      timestamp: segment.timestamp ?? null,
      speaker: segment.speaker ?? null,
      text: segment.text ?? '',
      matched_signals: matchedSignals,
      matched_phrases: matchedPhrases,
      lexicon_hits: lexiconHits,
      evidence_score: evidenceScore
    })
  }

  scoredSections.sort((left, right) => {
    if (right.evidence_score !== left.evidence_score) {
      return right.evidence_score - left.evidence_score
    }

    return left.timestampSeconds - right.timestampSeconds
  })

  return scoredSections.slice(0, MAX_EMPATHY_EVIDENCE_SECTIONS).map((section) => ({
    timestamp: section.timestamp,
    speaker: section.speaker,
    text: section.text,
    matched_signals: section.matched_signals,
    matched_phrases: section.matched_phrases,
    lexicon_hits: section.lexicon_hits,
    evidence_score: section.evidence_score
  }))
}

function analyzeEmpathyValidation(segments) {
  const understandingMatches = collectMatchedPhrases(segments, UNDERSTANDING_PATTERNS)
  const acknowledgementMatches = collectMatchedPhrases(segments, ACKNOWLEDGEMENT_PATTERNS)
  const empatheticPhraseMatches = collectMatchedPhrases(segments, EMPATHETIC_PHRASE_PATTERNS)
  const supportActionMatches = collectMatchedPhrases(segments, SUPPORT_ACTION_PATTERNS)

  let signalSegmentCount = 0
  let fullyQualifiedSegmentCount = 0
  let supportActionSegmentCount = 0

  for (const segment of segments) {
    const normalized = normalizeText(segment.text)
    const hasUnderstanding = segmentHasAnyPattern(normalized, UNDERSTANDING_PATTERNS)
    const hasAcknowledgement = segmentHasAnyPattern(normalized, ACKNOWLEDGEMENT_PATTERNS)
    const hasEmpatheticPhrase = segmentHasAnyPattern(normalized, EMPATHETIC_PHRASE_PATTERNS)
    const hasSupportAction = segmentHasAnyPattern(normalized, SUPPORT_ACTION_PATTERNS)

    if (hasUnderstanding || hasAcknowledgement || hasEmpatheticPhrase || hasSupportAction) {
      signalSegmentCount += 1
    }

    if (hasSupportAction) {
      supportActionSegmentCount += 1
    }

    if ((hasUnderstanding || hasAcknowledgement) && (hasEmpatheticPhrase || hasSupportAction)) {
      fullyQualifiedSegmentCount += 1
    }
  }

  const understandingCount = understandingMatches.size
  const acknowledgementCount = acknowledgementMatches.size
  const empatheticPhraseCount = empatheticPhraseMatches.size
  const supportActionCount = supportActionMatches.size
  const coreSignalCount = understandingCount + acknowledgementCount + empatheticPhraseCount
  const weightedScore =
    understandingCount * 2 +
    acknowledgementCount * 2 +
    empatheticPhraseCount * 2 +
    supportActionCount

  let recommendedBand = 0

  if (coreSignalCount === 0) {
    if (supportActionCount >= 2 && signalSegmentCount >= 2) {
      recommendedBand = 1
    }
  } else if (weightedScore >= 10 && fullyQualifiedSegmentCount >= 2 && signalSegmentCount >= 2) {
    recommendedBand = 3
  } else if (weightedScore >= 5 && signalSegmentCount >= 2) {
    recommendedBand = 2
  } else {
    recommendedBand = 1
  }

  return {
    hasUnderstanding: understandingCount > 0,
    hasAcknowledgement: acknowledgementCount > 0,
    hasEmpatheticPhrases: empatheticPhraseCount > 0,
    hasSupportActions: supportActionCount > 0,
    understandingCount,
    acknowledgementCount,
    empatheticPhraseCount,
    supportActionCount,
    coreSignalCount,
    weightedScore,
    recommendedBand,
    signalSegmentCount,
    supportActionSegmentCount,
    fullyQualifiedSegmentCount,
    understandingMatches,
    acknowledgementMatches,
    empatheticPhraseMatches,
    supportActionMatches,
    hasCoreSignal: coreSignalCount > 0
  }
}

function buildReason({
  selectedBand,
  counters,
  sentiment,
  matchedKeywords,
  validation,
  lexiconBand,
  lexiconHits,
  scoringSource
}) {
  const counterSummary = `Category counters -> C0:${counters[0]}, C1:${counters[1]}, C2:${counters[2]}, C3:${counters[3]}.`
  const sentimentSummary = `Sentiment signals -> positive:${sentiment.positiveHits}, negative:${sentiment.negativeHits}, net:${sentiment.sentimentScore}, suggested_band:${sentiment.band}.`
  const validationSummary =
    `Empathy signals -> acknowledgement:${validation.acknowledgementCount}, understanding:${validation.understandingCount}, empathetic_phrases:${validation.empatheticPhraseCount}, support_actions:${validation.supportActionCount}, weighted_score:${validation.weightedScore}.`
  const consistencySummary =
    `Consistency -> signal_segments:${validation.signalSegmentCount}, support_segments:${validation.supportActionSegmentCount}, qualified_segments:${validation.fullyQualifiedSegmentCount}.`

  if (selectedBand === 0) {
    return [
      'No reliable empathy evidence was found beyond neutral/procedural handling.',
      validationSummary,
      consistencySummary,
      counterSummary,
      sentimentSummary,
      'Final empathy score selected: 0.'
    ].join(' ')
  }

  const representativeBand = lexiconHits > 0 ? lexiconBand : selectedBand
  const topKeywords = [...(matchedKeywords[representativeBand] ?? [])].slice(0, 8).join(', ')
  const sourceSummary = `Scoring source: ${scoringSource}.`

  return [
    counterSummary,
    validationSummary,
    consistencySummary,
    sourceSummary,
    topKeywords ? `Representative matched tokens for selected category: ${topKeywords}.` : '',
    sentimentSummary
  ]
    .filter(Boolean)
    .join(' ')
}

function buildFailOutput(reason) {
  return {
    parameter: 'Empathy',
    empathy_score: 0,
    score: 0,
    status: 'BAND 0',
    category_counters: {
      category_0: 0,
      category_1: 0,
      category_2: 0,
      category_3: 0
    },
    sentiment: {
      positive_signals: 0,
      negative_signals: 0,
      net_score: 0,
      suggested_band: 1
    },
    empathy_validation: {
      passed: false,
      has_acknowledgement: false,
      has_understanding: false,
      has_empathetic_phrases: false,
      has_support_actions: false,
      acknowledgement_hits: 0,
      understanding_hits: 0,
      empathetic_phrase_hits: 0,
      support_action_hits: 0,
      core_signal_hits: 0,
      weighted_score: 0,
      recommended_band: 0,
      signal_segments: 0,
      support_action_segments: 0,
      fully_qualified_segments: 0
    },
    evidence_sections: [],
    reason
  }
}

export async function evaluateEmpathy(transcriptPayload) {
  const segments = extractTranscriptSegments(transcriptPayload)
  const csmSegments = segments.filter((segment) => isCsmSpeaker(segment.speaker))

  if (csmSegments.length === 0) {
    return buildFailOutput('No CSM segments found, so empathy could not be evaluated.')
  }

  let levelKeywordSets
  try {
    levelKeywordSets = await loadEmpathyLexicon()
  } catch {
    return buildFailOutput('Empathy dataset could not be loaded from Json/empathy_data.json.')
  }

  const { counters, matchedKeywords } = countEmpathyCategoryHits(csmSegments, levelKeywordSets)
  const sentiment = analyzeSentiment(csmSegments)
  const validation = analyzeEmpathyValidation(csmSegments)
  const evidenceSections = collectEmpathyEvidenceSections(csmSegments, levelKeywordSets)
  const totalCounterHits = counters[0] + counters[1] + counters[2] + counters[3]
  const lexiconSelection = totalCounterHits > 0 ? selectDominantBand(counters) : null
  const lexiconBand = lexiconSelection?.selectedBand ?? 0
  const lexiconMaxCount = lexiconSelection?.maxCount ?? 0

  let selectedBand = validation.recommendedBand
  let scoringSource = 'weighted'

  if (totalCounterHits > 0) {
    if (validation.hasCoreSignal && lexiconBand >= 2 && lexiconMaxCount >= 2) {
      selectedBand = Math.min(3, selectedBand + 1)
    }

    if (validation.hasCoreSignal && lexiconBand === 0 && validation.supportActionCount === 0) {
      selectedBand = Math.max(0, selectedBand - 1)
    }

    scoringSource = 'weighted+lexicon'
  }

  if (
    sentiment.negativeHits > 0 &&
    validation.acknowledgementCount === 0 &&
    validation.understandingCount <= 1 &&
    validation.empatheticPhraseCount <= 1
  ) {
    selectedBand = Math.max(0, selectedBand - 1)
  }

  if (sentiment.negativeHits >= sentiment.positiveHits + 2) {
    selectedBand = Math.min(selectedBand, 1)
  }

  if (
    validation.hasCoreSignal &&
    validation.signalSegmentCount >= 3 &&
    validation.supportActionCount >= 2 &&
    selectedBand === 1
  ) {
    selectedBand = 2
  }

  if (selectedBand >= 3 && validation.empatheticPhraseCount === 0) {
    selectedBand = 2
  }

  if (
    selectedBand >= 3 &&
    validation.empatheticPhraseCount < 2 &&
    validation.fullyQualifiedSegmentCount < 2
  ) {
    selectedBand = 2
  }

  if (!validation.hasCoreSignal && validation.supportActionCount < 2) {
    selectedBand = 0
  }

  return {
    parameter: 'Empathy',
    empathy_score: selectedBand,
    score: selectedBand,
    status: `BAND ${selectedBand}`,
    category_counters: {
      category_0: counters[0],
      category_1: counters[1],
      category_2: counters[2],
      category_3: counters[3]
    },
    sentiment: {
      positive_signals: sentiment.positiveHits,
      negative_signals: sentiment.negativeHits,
      net_score: sentiment.sentimentScore,
      suggested_band: sentiment.band
    },
    empathy_validation: {
      passed: selectedBand > 0,
      has_acknowledgement: validation.hasAcknowledgement,
      has_understanding: validation.hasUnderstanding,
      has_empathetic_phrases: validation.hasEmpatheticPhrases,
      has_support_actions: validation.hasSupportActions,
      acknowledgement_hits: validation.acknowledgementCount,
      understanding_hits: validation.understandingCount,
      empathetic_phrase_hits: validation.empatheticPhraseCount,
      support_action_hits: validation.supportActionCount,
      core_signal_hits: validation.coreSignalCount,
      weighted_score: validation.weightedScore,
      recommended_band: validation.recommendedBand,
      signal_segments: validation.signalSegmentCount,
      support_action_segments: validation.supportActionSegmentCount,
      fully_qualified_segments: validation.fullyQualifiedSegmentCount,
      matched_acknowledgement_phrases: [...validation.acknowledgementMatches],
      matched_understanding_phrases: [...validation.understandingMatches],
      matched_empathetic_phrases: [...validation.empatheticPhraseMatches],
      matched_support_action_phrases: [...validation.supportActionMatches]
    },
    evidence_sections: evidenceSections,
    reason: buildReason({
      selectedBand,
      counters,
      sentiment,
      matchedKeywords,
      validation,
      lexiconBand,
      lexiconHits: totalCounterHits,
      scoringSource
    })
  }
}
