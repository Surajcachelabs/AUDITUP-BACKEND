import fs from 'node:fs/promises'
import path from 'node:path'

const JSON_DIR = path.join(process.cwd(), 'Json')
const TIMESTAMP_LINE_REGEX =
  /^\s*["']?\[?(\d{1,2}(?::\d{2}){1,2}(?:\.\d+)?)\s*(?:-\s*(\d{1,2}(?::\d{2}){1,2}(?:\.\d+)?))?\]?\s*([A-Za-z][A-Za-z ]*)\s*:\s*(.*?)["']?\s*,?\s*$/

export function normalizeText(value) {
  return String(value ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export function parseTimestampToSeconds(timestampValue) {
  if (timestampValue == null) {
    return null
  }

  if (typeof timestampValue === 'number' && Number.isFinite(timestampValue)) {
    return timestampValue
  }

  const cleanTimestamp = String(timestampValue)
    .trim()
    .replace(/^\[/, '')
    .replace(/\]$/, '')

  if (!cleanTimestamp) {
    return null
  }

  const parts = cleanTimestamp.split(':').map(Number)
  if (parts.some((part) => Number.isNaN(part))) {
    return null
  }

  if (parts.length === 1) {
    return parts[0]
  }

  if (parts.length === 2) {
    const [minutes, seconds] = parts
    return minutes * 60 + seconds
  }

  if (parts.length === 3) {
    const [hours, minutes, seconds] = parts
    return hours * 3600 + minutes * 60 + seconds
  }

  return null
}

export function formatTimestampFromSeconds(value) {
  const wholeSeconds = Math.max(0, Math.floor(Number(value) || 0))
  const hours = Math.floor(wholeSeconds / 3600)
  const minutes = Math.floor((wholeSeconds % 3600) / 60)
  const seconds = wholeSeconds % 60

  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
  }

  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

export function parseTranscriptInput(payload) {
  if (typeof payload === 'string') {
    return payload
  }

  const collected = []

  const walk = (value) => {
    if (value == null) {
      return
    }

    if (typeof value === 'string') {
      collected.push(value)
      return
    }

    if (Array.isArray(value)) {
      value.forEach(walk)
      return
    }

    if (typeof value === 'object') {
      Object.values(value).forEach(walk)
    }
  }

  walk(payload)
  return collected.join('\n')
}

function parseSegmentsFromText(transcriptText) {
  const lines = String(transcriptText ?? '').split(/\r?\n/)
  const segments = []

  for (const line of lines) {
    const match = line.match(TIMESTAMP_LINE_REGEX)
    if (!match) {
      continue
    }

    const [, startTimestampRaw, endTimestampRaw, speakerRaw, textRaw] = match
    const timestampSeconds = parseTimestampToSeconds(startTimestampRaw)
    const endTimestampSeconds = parseTimestampToSeconds(endTimestampRaw) ?? timestampSeconds

    if (timestampSeconds == null) {
      continue
    }

    segments.push({
      timestamp: String(startTimestampRaw).trim(),
      timestampSeconds,
      endTimestampSeconds,
      speaker: String(speakerRaw).trim(),
      text: String(textRaw).trim()
    })
  }

  return segments
}

function getObjectValueByKeys(item, keys) {
  for (const key of keys) {
    if (item[key] != null) {
      return item[key]
    }
  }
  return null
}

function parseSegmentsFromStructuredPayload(payload) {
  const segments = []

  const visit = (value) => {
    if (value == null) {
      return
    }

    if (Array.isArray(value)) {
      value.forEach(visit)
      return
    }

    if (typeof value !== 'object') {
      return
    }

    const textValue = getObjectValueByKeys(value, [
      'text',
      'spoken_text',
      'utterance',
      'content',
      'message',
      'transcript'
    ])
    const speakerValue = getObjectValueByKeys(value, ['speaker', 'role', 'participant'])
    const timestampValue = getObjectValueByKeys(value, [
      'timestamp',
      'time',
      'start_time',
      'startTime',
      'offset_seconds'
    ])

    const timestampSeconds = parseTimestampToSeconds(timestampValue)

    if (
      typeof textValue === 'string' &&
      typeof speakerValue === 'string' &&
      timestampSeconds != null
    ) {
      segments.push({
        timestamp:
          typeof timestampValue === 'string'
            ? String(timestampValue).trim()
            : formatTimestampFromSeconds(timestampSeconds),
        timestampSeconds,
        endTimestampSeconds: timestampSeconds,
        speaker: String(speakerValue).trim(),
        text: textValue.trim()
      })
    }

    Object.values(value).forEach(visit)
  }

  visit(payload)
  return segments
}

export function extractTranscriptSegments(payload) {
  if (typeof payload === 'string') {
    return parseSegmentsFromText(payload)
  }

  const structuredSegments = parseSegmentsFromStructuredPayload(payload)
  if (structuredSegments.length > 0) {
    return structuredSegments
  }

  return parseSegmentsFromText(parseTranscriptInput(payload))
}

export function isCsmSpeaker(speaker) {
  const normalizedSpeaker = normalizeText(speaker).replace(/\s+/g, '')
  return normalizedSpeaker === 'csm' || normalizedSpeaker.startsWith('csm')
}

export function getTotalCallDurationSeconds(segments) {
  if (!segments || segments.length === 0) {
    return 0
  }

  return Math.max(
    ...segments.map((segment) =>
      typeof segment.endTimestampSeconds === 'number'
        ? segment.endTimestampSeconds
        : segment.timestampSeconds
    )
  )
}

function parseMaybeBrokenJson(rawText) {
  const trimmed = rawText.trim()

  try {
    return JSON.parse(trimmed)
  } catch {
    const firstObject = trimmed.indexOf('{')
    const firstArray = trimmed.indexOf('[')

    const start = [firstObject, firstArray]
      .filter((value) => value >= 0)
      .sort((a, b) => a - b)[0]

    if (start == null) {
      throw new Error('JSON dictionary cannot be parsed')
    }

    return JSON.parse(trimmed.slice(start))
  }
}

function extractPhrasesFromDictionary(dictionaryData) {
  if (Array.isArray(dictionaryData)) {
    return dictionaryData
      .flatMap((item) => {
        if (typeof item === 'string') {
          return [item]
        }

        if (item && typeof item === 'object') {
          return Object.values(item).filter((value) => typeof value === 'string')
        }

        return []
      })
      .map((value) => value.trim())
      .filter(Boolean)
  }

  if (dictionaryData && typeof dictionaryData === 'object') {
    return Object.values(dictionaryData)
      .flatMap((value) => {
        if (Array.isArray(value)) {
          return value
        }

        if (typeof value === 'string') {
          return [value]
        }

        return []
      })
      .map((value) => String(value).trim())
      .filter(Boolean)
  }

  return []
}

export async function loadDictionaryPhrases(fileNameOrNames) {
  const candidateFiles = Array.isArray(fileNameOrNames) ? fileNameOrNames : [fileNameOrNames]
  let lastReadError = null

  for (const fileName of candidateFiles) {
    const filePath = path.join(JSON_DIR, fileName)

    try {
      const rawText = await fs.readFile(filePath, 'utf8')
      const data = parseMaybeBrokenJson(rawText)
      return extractPhrasesFromDictionary(data)
    } catch (error) {
      lastReadError = error
    }
  }

  throw lastReadError ?? new Error('Failed to load dictionary phrases')
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

export function matchSegmentsAgainstPhrases(segments, repositoryPhrases) {
  const phraseRecords = repositoryPhrases
    .map((phrase) => {
      const normalized = normalizeText(phrase)
      const tokenCount = normalized.split(' ').filter(Boolean).length

      return {
        phrase,
        normalized,
        tokenCount
      }
    })
    .filter((entry) => entry.normalized)

  for (const segment of segments) {
    const normalizedSegment = normalizeText(segment.text)

    for (const phraseRecord of phraseRecords) {
      const strictMatch =
        normalizedSegment.includes(phraseRecord.normalized) ||
        (phraseRecord.tokenCount === 1 &&
          normalizedSegment.split(' ').includes(phraseRecord.normalized))

      if (strictMatch) {
        return {
          matched: true,
          matchedPhrase: phraseRecord.phrase,
          timestamp: segment.timestamp,
          mode: 'repository',
          confidence: 1
        }
      }
    }
  }

  let bestCandidate = null

  for (const segment of segments) {
    const normalizedSegment = normalizeText(segment.text)

    for (const phraseRecord of phraseRecords) {
      const phraseCoverage = computePhraseCoverage(normalizedSegment, phraseRecord.normalized)
      const ngramScore = jaccardSimilarity(normalizedSegment, phraseRecord.normalized)
      const score = Math.max(phraseCoverage, ngramScore)

      const threshold = phraseRecord.tokenCount <= 2 ? 0.8 : phraseRecord.tokenCount <= 5 ? 0.7 : 0.6

      if (score >= threshold) {
        if (!bestCandidate || score > bestCandidate.score) {
          bestCandidate = {
            score,
            phrase: phraseRecord.phrase,
            timestamp: segment.timestamp
          }
        }
      }
    }
  }

  if (bestCandidate) {
    return {
      matched: true,
      matchedPhrase: bestCandidate.phrase,
      timestamp: bestCandidate.timestamp,
      mode: 'flexible',
      confidence: Number(bestCandidate.score.toFixed(3))
    }
  }

  return {
    matched: false,
    matchedPhrase: null,
    timestamp: null,
    mode: null,
    confidence: 0
  }
}
