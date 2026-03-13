import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { extractTranscriptSegments, normalizeText } from './shared.js'

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url))
const SLANG_DATA_PATH = path.join(MODULE_DIR, '..', 'Json', 'slang_severity_data.json')
const MIN_MATCH_SIMILARITY = 0.7
const WINDOW_SIZE_DELTA = 2

let cachedPhraseWeights = null

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function toSafeWeight(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return null
  }

  return numeric
}

function tokenize(value) {
  const normalized = normalizeText(value)
  return normalized ? normalized.split(' ') : []
}

function toRounded(value, digits = 2) {
  return Number(value.toFixed(digits))
}

function toCeiledScore(value) {
  return Math.max(0, Math.min(5, Math.ceil(value)))
}

function levenshteinDistance(left, right) {
  if (left === right) {
    return 0
  }

  if (!left.length) {
    return right.length
  }

  if (!right.length) {
    return left.length
  }

  const previous = Array.from({ length: right.length + 1 }, (_, index) => index)
  const current = new Array(right.length + 1)

  for (let i = 0; i < left.length; i += 1) {
    current[0] = i + 1

    for (let j = 0; j < right.length; j += 1) {
      const cost = left[i] === right[j] ? 0 : 1
      current[j + 1] = Math.min(current[j] + 1, previous[j + 1] + 1, previous[j] + cost)
    }

    for (let j = 0; j < previous.length; j += 1) {
      previous[j] = current[j]
    }
  }

  return previous[right.length]
}

function computeStringSimilarity(left, right) {
  const maxLength = Math.max(left.length, right.length)
  if (maxLength === 0) {
    return 1
  }

  return 1 - levenshteinDistance(left, right) / maxLength
}

function selectNonOverlappingMatches(candidates) {
  const sorted = [...candidates].sort((left, right) => {
    if (right.similarity !== left.similarity) {
      return right.similarity - left.similarity
    }

    if (left.startIndex !== right.startIndex) {
      return left.startIndex - right.startIndex
    }

    return left.endIndex - right.endIndex
  })

  const selected = []

  for (const candidate of sorted) {
    const overlaps = selected.some(
      (existing) => candidate.startIndex <= existing.endIndex && candidate.endIndex >= existing.startIndex
    )

    if (!overlaps) {
      selected.push(candidate)
    }
  }

  return selected.sort((left, right) => left.startIndex - right.startIndex)
}

function scoreCandidateSimilarity(phraseText, candidateTokens) {
  const candidateText = candidateTokens.join(' ')
  return {
    candidateText,
    similarity: computeStringSimilarity(phraseText, candidateText)
  }
}

function findPhraseMatchesInSegment(segmentText, phraseEntry) {
  const normalizedSegmentText = normalizeText(segmentText)
  const exactRegex = new RegExp(`\\b${escapeRegExp(phraseEntry.phrase)}\\b`, 'g')
  const exactMatches = [...normalizedSegmentText.matchAll(exactRegex)]

  if (exactMatches.length > 0) {
    return exactMatches.map(() => ({
      candidateText: phraseEntry.phrase,
      similarity: 1,
      startIndex: 0,
      endIndex: 0
    }))
  }

  const segmentTokens = tokenize(segmentText)

  if (segmentTokens.length === 0 || phraseEntry.phraseTokens.length === 0) {
    return []
  }

  const candidates = []
  const minWindowSize = Math.max(1, phraseEntry.phraseTokens.length - WINDOW_SIZE_DELTA)
  const maxWindowSize = Math.min(segmentTokens.length, phraseEntry.phraseTokens.length + WINDOW_SIZE_DELTA)

  for (let windowSize = minWindowSize; windowSize <= maxWindowSize; windowSize += 1) {
    for (let startIndex = 0; startIndex <= segmentTokens.length - windowSize; startIndex += 1) {
      const endIndex = startIndex + windowSize - 1
      const candidateTokens = segmentTokens.slice(startIndex, endIndex + 1)
      const scoredCandidate = scoreCandidateSimilarity(phraseEntry.phrase, candidateTokens)

      if (scoredCandidate.similarity >= MIN_MATCH_SIMILARITY) {
        candidates.push({
          ...scoredCandidate,
          startIndex,
          endIndex
        })
      }
    }
  }

  return selectNonOverlappingMatches(candidates)
}

async function loadSlangPhraseWeights() {
  if (cachedPhraseWeights) {
    return cachedPhraseWeights
  }

  const rawText = await fs.readFile(SLANG_DATA_PATH, 'utf8')
  const parsed = JSON.parse(rawText)
  const rows = Array.isArray(parsed)
    ? parsed
    : Array.isArray(parsed?.['SlangInformal Language'])
      ? parsed['SlangInformal Language']
      : null

  if (!rows) {
    throw new Error('Invalid slang_severity_data.json format. Expected array or SlangInformal Language array.')
  }

  const merged = new Map()

  for (const row of rows) {
    const phrase = normalizeText(row?.Phrase)
    const weight = toSafeWeight(row?.['Base Severity Weight (0-5)'] ?? row?.['Base Severity Weight'])

    if (!phrase || weight == null) {
      continue
    }

    const phraseTokens = tokenize(phrase)
    const existing = merged.get(phrase)

    if (existing == null || weight > existing.weight) {
      merged.set(phrase, {
        phrase,
        phraseTokens,
        weight
      })
    }
  }

  cachedPhraseWeights = [...merged.values()]
  return cachedPhraseWeights
}

function toChunkLabel(chunkIndex, startLine, endLine) {
  return `Chunk ${chunkIndex + 1} (lines ${startLine}-${endLine})`
}

function determineChunkCount(lineCount) {
  if (lineCount < 100) return 5
  if (lineCount < 150) return 7
  if (lineCount < 200) return 10
  if (lineCount < 250) return 12
  if (lineCount < 300) return 15
  if (lineCount < 400) return 20
  if (lineCount < 500) return 25
  return Math.ceil(lineCount / 20)
}

function buildFailOutput(reason) {
  return {
    parameter: 'Slang Severity',
    slang_score: 5,
    impact_score: 5,
    score: 5,
    points: 5,
    final_value: 5,
    max_counter: 0,
    final_impact_value: 0,
    max_impact_value: 0,
    trigger_value: 0,
    total_impact_value: 0,
    section_count: 0,
    detected_section_count: 0,
    section_analysis: [],
    no_slang_detected: true,
    status: 'POINTS 5',
    reason
  }
}

function evaluateChunkPhrases(chunkSegments, phraseWeights) {
  const matchedPhrases = []
  let selectedPhrase = null

  for (const entry of phraseWeights) {
    const matches = []

    for (const segment of chunkSegments) {
      const segmentMatches = findPhraseMatchesInSegment(segment.text, entry)

      for (const match of segmentMatches) {
        matches.push({
          matched_text: match.candidateText,
          matching_confidence: toRounded(match.similarity, 4),
          timestamp: segment.timestamp ?? null,
          speaker: segment.speaker ?? null
        })
      }
    }

    const frequency = matches.length

    if (frequency === 0) {
      continue
    }

    const impactValue = entry.weight * frequency
    const matchingConfidence = Math.max(...matches.map((match) => match.matching_confidence))

    const phraseResult = {
      phrase: entry.phrase,
      base_severity_weight: entry.weight,
      frequency,
      impact_value: impactValue,
      matching_confidence: matchingConfidence,
      matches
    }

    matchedPhrases.push(phraseResult)

    const isBetterChoice =
      selectedPhrase == null ||
      phraseResult.impact_value > selectedPhrase.impact_value ||
      (phraseResult.impact_value === selectedPhrase.impact_value &&
        phraseResult.matching_confidence > selectedPhrase.matching_confidence) ||
      (phraseResult.impact_value === selectedPhrase.impact_value &&
        phraseResult.matching_confidence === selectedPhrase.matching_confidence &&
        phraseResult.base_severity_weight > selectedPhrase.base_severity_weight)

    if (isBetterChoice) {
      selectedPhrase = phraseResult
    }
  }

  const chunkImpact = selectedPhrase?.impact_value ?? 0
  const triggerValue = chunkImpact / 5
  const sectionScore = toCeiledScore(5 - triggerValue)

  return {
    matchedPhrases,
    selectedPhrase,
    chunkImpact,
    triggerValue,
    sectionScore
  }
}

export async function evaluateSlangSeverity(transcriptPayload) {
  const segments = extractTranscriptSegments(transcriptPayload)

  if (segments.length === 0) {
    return buildFailOutput('No timestamped transcript segments were detected; slang severity defaults to zero impact.')
  }

  let phraseWeights
  try {
    phraseWeights = await loadSlangPhraseWeights()
  } catch {
    return buildFailOutput('Slang severity dataset could not be loaded from Json/slang_severity_data.json.')
  }

  if (phraseWeights.length === 0) {
    return buildFailOutput('Slang severity dataset has no valid phrase entries.')
  }

  const lineCount = segments.length
  const chunkCount = Math.max(1, determineChunkCount(lineCount))
  const linesPerChunk = Math.ceil(lineCount / chunkCount)

  const detectedSectionAnalysis = []
  let maxCounter = 0
  let totalImpactValue = 0

  for (let index = 0; index < chunkCount; index += 1) {
    const startIdx = index * linesPerChunk
    const endIdx = Math.min(startIdx + linesPerChunk, lineCount)

    if (startIdx >= lineCount) {
      break
    }

    const chunkSegments = segments.slice(startIdx, endIdx)
    const chunkEvaluation = evaluateChunkPhrases(chunkSegments, phraseWeights)

    totalImpactValue += chunkEvaluation.chunkImpact
    maxCounter = Math.max(maxCounter, chunkEvaluation.chunkImpact)

    if (chunkEvaluation.chunkImpact > 0 && chunkEvaluation.selectedPhrase) {
      detectedSectionAnalysis.push({
        section: toChunkLabel(index, startIdx + 1, endIdx),
        selected_phrase: chunkEvaluation.selectedPhrase.phrase,
        matched_phrases: chunkEvaluation.matchedPhrases,
        impact_value: toRounded(chunkEvaluation.chunkImpact),
        trigger_value: toRounded(chunkEvaluation.triggerValue),
        section_score: toRounded(chunkEvaluation.sectionScore),
        matching_confidence: chunkEvaluation.selectedPhrase.matching_confidence
      })
    }
  }

  const finalImpactValue = maxCounter
  const triggerValue = finalImpactValue / 5
  const roundedScore = toCeiledScore(5 - triggerValue)
  const noSlangDetected = roundedScore === 5 && detectedSectionAnalysis.length === 0

  return {
    parameter: 'Slang Severity',
    slang_score: roundedScore,
    impact_score: roundedScore,
    score: roundedScore,
    points: roundedScore,
    final_value: roundedScore,
    max_counter: toRounded(maxCounter),
    final_impact_value: toRounded(finalImpactValue),
    max_impact_value: toRounded(finalImpactValue),
    trigger_value: toRounded(triggerValue),
    total_impact_value: toRounded(totalImpactValue),
    section_count: Math.ceil(lineCount / linesPerChunk),
    detected_section_count: detectedSectionAnalysis.length,
    section_analysis: detectedSectionAnalysis,
    no_slang_detected: noSlangDetected,
    status: `POINTS ${roundedScore}`,
    reason:
      `Each chunk is compared against all dataset phrases using simple string similarity. ` +
      `Only matches with at least ${(MIN_MATCH_SIMILARITY * 100).toFixed(0)}% similarity are counted. ` +
      `For each chunk, the highest phrase impact (frequency × base severity weight) becomes the chunk impact. ` +
      `The highest chunk impact across the call becomes the final impact value. ` +
      `Trigger value = final impact / 5, and final score = ceiling(5 - trigger value) = ${roundedScore}.`
  }
}
