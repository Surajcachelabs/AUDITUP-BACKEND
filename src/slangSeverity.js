import fs from 'node:fs/promises'
import path from 'node:path'
import {
  extractTranscriptSegments,
  formatTimestampFromSeconds,
  getTotalCallDurationSeconds,
  normalizeText
} from './shared.js'

const SLANG_DATA_PATH = path.join(process.cwd(), 'Json', 'slang_severity_data.json')
const SECTION_SECONDS = 180
const MAX_SLANG_SCORE = 5

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

async function loadSlangPhraseWeights() {
  if (cachedPhraseWeights) {
    return cachedPhraseWeights
  }

  const rawText = await fs.readFile(SLANG_DATA_PATH, 'utf8')
  const parsed = JSON.parse(rawText)

  if (!Array.isArray(parsed)) {
    throw new Error('Invalid slang_severity_data.json format. Expected array.')
  }

  const merged = new Map()

  for (const row of parsed) {
    const phrase = normalizeText(row?.Phrase)
    const weight = toSafeWeight(row?.['Base Severity Weight'])

    if (!phrase || weight == null) {
      continue
    }

    const existing = merged.get(phrase)

    // Keep the strictest weight when duplicated phrases exist.
    if (existing == null || weight > existing) {
      merged.set(phrase, weight)
    }
  }

  cachedPhraseWeights = [...merged.entries()].map(([phrase, weight]) => ({ phrase, weight }))
  return cachedPhraseWeights
}

function countPhraseInText(sectionText, phrase) {
  const regex = new RegExp(`\\b${escapeRegExp(phrase)}\\b`, 'g')
  const matches = sectionText.match(regex)
  return matches ? matches.length : 0
}

function chooseSectionTopPhrase(sectionText, phraseWeights) {
  let best = {
    phrase: null,
    weight: 0,
    frequency: 0,
    impactValue: 0
  }

  for (const entry of phraseWeights) {
    const frequency = countPhraseInText(sectionText, entry.phrase)

    if (frequency === 0) {
      continue
    }

    const impactValue = frequency * entry.weight

    const isBetter =
      frequency > best.frequency ||
      (frequency === best.frequency && entry.weight > best.weight) ||
      (frequency === best.frequency && entry.weight === best.weight && impactValue > best.impactValue)

    if (isBetter) {
      best = {
        phrase: entry.phrase,
        weight: entry.weight,
        frequency,
        impactValue
      }
    }
  }

  return best
}

function toSectionLabel(startSeconds, endSeconds) {
  return `${formatTimestampFromSeconds(startSeconds)} - ${formatTimestampFromSeconds(endSeconds)}`
}

function mapImpactToScore(averageImpactValue) {
  if (averageImpactValue <= 0) {
    return 5
  }

  if (averageImpactValue <= 3) {
    return 4
  }

  if (averageImpactValue <= 6) {
    return 3
  }

  if (averageImpactValue <= 9) {
    return 2
  }

  if (averageImpactValue <= 14) {
    return 1
  }

  return 0
}

function buildFailOutput(reason) {
  return {
    parameter: 'Slang Severity',
    slang_score: 5,
    score: 5,
    final_value: 0,
    average_impact_value: 0,
    total_impact_value: 0,
    section_count: 0,
    section_analysis: [],
    status: 'NO DATA',
    reason
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

  const sortedSegments = [...segments].sort((a, b) => a.timestampSeconds - b.timestampSeconds)
  const totalDurationSeconds = Math.max(getTotalCallDurationSeconds(sortedSegments), 1)
  const sectionCount = Math.max(1, Math.ceil(totalDurationSeconds / SECTION_SECONDS))

  const sectionAnalysis = []
  let totalImpactValue = 0

  for (let index = 0; index < sectionCount; index += 1) {
    const startSeconds = index * SECTION_SECONDS
    const endSeconds = Math.min((index + 1) * SECTION_SECONDS, totalDurationSeconds)

    const sectionSegments = sortedSegments.filter((segment) => {
      if (index === sectionCount - 1) {
        return segment.timestampSeconds >= startSeconds && segment.timestampSeconds <= endSeconds
      }

      return segment.timestampSeconds >= startSeconds && segment.timestampSeconds < endSeconds
    })

    const sectionText = normalizeText(sectionSegments.map((segment) => segment.text).join(' '))
    const top = chooseSectionTopPhrase(sectionText, phraseWeights)

    totalImpactValue += top.impactValue

    sectionAnalysis.push({
      section: toSectionLabel(startSeconds, endSeconds),
      selected_phrase: top.phrase,
      base_severity_weight: top.weight,
      frequency: top.frequency,
      impact_value: top.impactValue
    })
  }

  const averageImpactValue = sectionCount === 0 ? 0 : totalImpactValue / sectionCount
  const slangScore = mapImpactToScore(averageImpactValue)

  // Final value is the deduction amount requested for total score subtraction.
  const finalValue = MAX_SLANG_SCORE - slangScore

  return {
    parameter: 'Slang Severity',
    slang_score: slangScore,
    score: slangScore,
    final_value: finalValue,
    average_impact_value: Number(averageImpactValue.toFixed(2)),
    total_impact_value: Number(totalImpactValue.toFixed(2)),
    section_count: sectionCount,
    section_analysis: sectionAnalysis,
    status: `DEDUCTION ${finalValue}`,
    reason:
      `Average impact value is ${averageImpactValue.toFixed(2)} across ${sectionCount} sections (3-minute windows). ` +
      `Impact per section is computed as highest matched phrase frequency multiplied by its base severity weight. ` +
      `Mapped slang score is ${slangScore} using configured impact bands, so final subtraction value is ${finalValue}.`
  }
}
