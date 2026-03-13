import OpenAI from 'openai'

let cachedApiKey = null
let cachedClient = null

function getOpenAiClient() {
  const apiKey = process.env.OPENAI_API_KEY

  if (!apiKey) {
    return null
  }

  if (cachedClient && cachedApiKey === apiKey) {
    return cachedClient
  }

  cachedApiKey = apiKey
  cachedClient = new OpenAI({ apiKey })
  return cachedClient
}

function parseJsonFromText(content) {
  if (!content || typeof content !== 'string') {
    return null
  }

  try {
    return JSON.parse(content)
  } catch {
    const firstCurly = content.indexOf('{')
    const lastCurly = content.lastIndexOf('}')

    if (firstCurly < 0 || lastCurly < 0 || lastCurly <= firstCurly) {
      return null
    }

    try {
      return JSON.parse(content.slice(firstCurly, lastCurly + 1))
    } catch {
      return null
    }
  }
}

function toNonEmptyString(value) {
  if (typeof value !== 'string') {
    return null
  }

  const trimmed = value.trim()
  return trimmed ? trimmed : null
}

export function segmentsToTranscriptText(segments) {
  return segments
    .map((segment) => `${segment.timestamp} ${segment.speaker}: ${segment.text}`)
    .join('\n')
}

export async function runOpenAiIntentFallback({ parameterName, instructions, segments }) {
  if (!segments || segments.length === 0) {
    return null
  }

  const client = getOpenAiClient()
  if (!client) {
    return null
  }

  const model = process.env.OPENAI_MODEL || 'gpt-5.3-codex'
  const transcriptText = segmentsToTranscriptText(segments)

  const systemPrompt = [
    `You are a deterministic ${parameterName} evaluator.`,
    'You must return JSON only.',
    'No markdown. No code fences.',
    'Output JSON schema:',
    '{"detected": boolean, "timestamp": string|null, "text": string|null, "reason": string}'
  ].join('\n')

  const userPrompt = [
    instructions,
    '',
    'Transcript to evaluate:',
    transcriptText
  ].join('\n')

  try {
    const completion = await client.chat.completions.create({
      model,
      temperature: 0,
      messages: [
        {
          role: 'system',
          content: systemPrompt
        },
        {
          role: 'user',
          content: userPrompt
        }
      ]
    })

    const content = completion.choices?.[0]?.message?.content
    const parsed = parseJsonFromText(content)

    if (!parsed || typeof parsed.detected !== 'boolean') {
      return null
    }

    return {
      detected: parsed.detected,
      timestamp: typeof parsed.timestamp === 'string' ? parsed.timestamp : null,
      text: typeof parsed.text === 'string' ? parsed.text : null,
      reason: typeof parsed.reason === 'string' ? parsed.reason : ''
    }
  } catch {
    return null
  }
}

export async function runOpenAiSlangIntentCheck({ phrase, candidateText, chunkText }) {
  const normalizedPhrase = toNonEmptyString(phrase)
  const normalizedCandidateText = toNonEmptyString(candidateText)
  const normalizedChunkText = toNonEmptyString(chunkText)

  if (!normalizedPhrase || !normalizedCandidateText || !normalizedChunkText) {
    return null
  }

  const client = getOpenAiClient()
  if (!client) {
    return null
  }

  const model = process.env.OPENAI_MODEL || 'gpt-5.3-codex'

  const systemPrompt = [
    'You are a strict slang intent verifier for call audit scoring.',
    'Return JSON only.',
    'No markdown. No code fences.',
    'Output JSON schema:',
    '{"intent_match": boolean, "confidence": number, "reason": string}'
  ].join('\n')

  const userPrompt = [
    'Determine whether the candidate text is actually being used with the same offensive, dismissive, or slang intent as the target phrase.',
    'Be strict. Do not approve weak or ambiguous paraphrases.',
    '',
    `Target phrase: ${normalizedPhrase}`,
    `Candidate text: ${normalizedCandidateText}`,
    `Chunk context: ${normalizedChunkText}`
  ].join('\n')

  try {
    const completion = await client.chat.completions.create({
      model,
      temperature: 0,
      messages: [
        {
          role: 'system',
          content: systemPrompt
        },
        {
          role: 'user',
          content: userPrompt
        }
      ]
    })

    const content = completion.choices?.[0]?.message?.content
    const parsed = parseJsonFromText(content)

    if (!parsed || typeof parsed.intent_match !== 'boolean') {
      return null
    }

    const confidence = Number(parsed.confidence)

    return {
      intentMatch: parsed.intent_match,
      confidence: Number.isFinite(confidence) ? confidence : 0,
      reason: typeof parsed.reason === 'string' ? parsed.reason : ''
    }
  } catch {
    return null
  }
}

export async function runOpenAiEmpathyStressDetection({ segments }) {
  if (!segments || segments.length === 0) {
    return null
  }

  const client = getOpenAiClient()
  if (!client) {
    return null
  }

  const model = process.env.OPENAI_MODEL || 'gpt-5.3-codex'
  const indexedTranscript = segments
    .map((segment, index) => `[${index}] ${segment.timestamp ?? 'NA'} ${segment.speaker}: ${segment.text}`)
    .join('\n')

  const systemPrompt = [
    'You identify client stress signals in a call transcript for empathy evaluation.',
    'Return JSON only.',
    'No markdown. No code fences.',
    'Only select transcript indexes where the CLIENT appears stressed, frustrated, confused, anxious, concerned, or worried.',
    'Output JSON schema:',
    '{"stress_segments": [{"index": number, "emotion": string, "reason": string}]}'
  ].join('\n')

  const userPrompt = [
    'Review the full transcript and identify only client utterances that clearly express stress, frustration, confusion, concern, urgency, or emotional difficulty.',
    'Do not include CSM lines.',
    'Use only the provided transcript indexes.',
    '',
    'Transcript:',
    indexedTranscript
  ].join('\n')

  try {
    const completion = await client.chat.completions.create({
      model,
      temperature: 0,
      messages: [
        {
          role: 'system',
          content: systemPrompt
        },
        {
          role: 'user',
          content: userPrompt
        }
      ]
    })

    const content = completion.choices?.[0]?.message?.content
    const parsed = parseJsonFromText(content)
    const items = Array.isArray(parsed?.stress_segments) ? parsed.stress_segments : null

    if (!items) {
      return null
    }

    return items
      .map((item) => ({
        index: Number(item?.index),
        emotion: typeof item?.emotion === 'string' ? item.emotion.trim() : '',
        reason: typeof item?.reason === 'string' ? item.reason.trim() : ''
      }))
      .filter((item) => Number.isInteger(item.index) && item.index >= 0 && item.index < segments.length)
  } catch {
    return null
  }
}

export async function runOpenAiStructuredSummary({ segments }) {
  if (!segments || segments.length === 0) {
    return null
  }

  const client = getOpenAiClient()
  if (!client) {
    return null
  }

  const model = process.env.OPENAI_MODEL || 'gpt-5.3-codex'
  const transcriptText = segmentsToTranscriptText(segments)

  const systemPrompt = [
    'You are a deterministic conversation summarizer for CSM-client calls.',
    'Use the full transcript context, not just closing lines.',
    'Return JSON only. No markdown. No code fences.',
    'Output JSON schema:',
    '{"customer_issue": string, "csm_actions": string, "resolution_next_steps": string, "final_outcome": string, "overall_summary": string, "reason": string}'
  ].join('\n')

  const userPrompt = [
    'Summarize this conversation across the full interaction flow.',
    'Capture these intents explicitly:',
    '1) customer_issue/problem',
    '2) actions taken by CSM',
    '3) resolution or next steps',
    '4) final outcome',
    'Keep each field concise and factual, grounded in transcript evidence.',
    '',
    'Transcript to summarize:',
    transcriptText
  ].join('\n')

  try {
    const completion = await client.chat.completions.create({
      model,
      temperature: 0,
      messages: [
        {
          role: 'system',
          content: systemPrompt
        },
        {
          role: 'user',
          content: userPrompt
        }
      ]
    })

    const content = completion.choices?.[0]?.message?.content
    const parsed = parseJsonFromText(content)

    if (!parsed || typeof parsed !== 'object') {
      return null
    }

    const customerIssue = toNonEmptyString(parsed.customer_issue)
    const csmActions = toNonEmptyString(parsed.csm_actions)
    const resolutionNextSteps = toNonEmptyString(parsed.resolution_next_steps)
    const finalOutcome = toNonEmptyString(parsed.final_outcome)
    const overallSummary = toNonEmptyString(parsed.overall_summary)
    const reason = toNonEmptyString(parsed.reason)

    const fieldCoverage = [customerIssue, csmActions, resolutionNextSteps, finalOutcome].filter(Boolean)
      .length

    if (fieldCoverage === 0) {
      return null
    }

    return {
      customerIssue,
      csmActions,
      resolutionNextSteps,
      finalOutcome,
      overallSummary,
      reason,
      fieldCoverage
    }
  } catch {
    return null
  }
}
