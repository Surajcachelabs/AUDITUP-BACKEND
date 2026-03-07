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

function buildReason({
  fromSentimentOnly,
  selectedBand,
  counters,
  maxCount,
  sentiment,
  matchedKeywords
}) {
  const counterSummary = `Category counters -> C0:${counters[0]}, C1:${counters[1]}, C2:${counters[2]}, C3:${counters[3]}.`
  const sentimentSummary = `Sentiment signals -> positive:${sentiment.positiveHits}, negative:${sentiment.negativeHits}, net:${sentiment.sentimentScore}, suggested_band:${sentiment.band}.`

  if (fromSentimentOnly) {
    return [
      'No direct empathy lexicon token matches were found in the transcript, so score falls back to sentiment analysis.',
      sentimentSummary,
      `Final empathy score selected: ${selectedBand}.`
    ].join(' ')
  }

  const topKeywords = [...matchedKeywords[selectedBand]].slice(0, 8).join(', ')
  const tieAwareSummary =
    maxCount > 0
      ? `Dominant category selected from counters: ${selectedBand} (highest hit count ${maxCount}; ties resolved toward higher category).`
      : `All counters are zero; selected category ${selectedBand} by priority rule.`

  return [
    counterSummary,
    tieAwareSummary,
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
  const totalCounterHits = counters[0] + counters[1] + counters[2] + counters[3]

  let selectedBand
  let maxCount
  let fromSentimentOnly = false

  if (totalCounterHits === 0) {
    selectedBand = sentiment.band
    maxCount = 0
    fromSentimentOnly = true
  } else {
    const selection = selectDominantBand(counters)
    selectedBand = selection.selectedBand
    maxCount = selection.maxCount
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
    reason: buildReason({
      fromSentimentOnly,
      selectedBand,
      counters,
      maxCount,
      sentiment,
      matchedKeywords
    })
  }
}
