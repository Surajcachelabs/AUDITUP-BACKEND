import {
  extractTranscriptSegments,
  getTotalCallDurationSeconds,
  isCsmSpeaker,
  loadDictionaryPhrases,
  matchSegmentsAgainstPhrases
} from './shared.js'
import { runOpenAiIntentFallback } from './openAiFallback.js'

function buildGreetingOutput({ detected, matchedPhrase = null, timestamp = null, reason }) {
  return {
    parameter: 'Greeting Detection',
    greeting_detected: detected ? 'Yes' : 'No',
    score: detected ? 1 : 0,
    matched_phrase: detected ? matchedPhrase : null,
    timestamp: detected ? timestamp : null,
    speaker: detected ? 'CSM' : null,
    reason
  }
}

function withMeta(result, decisionSource, options) {
  if (options?.returnMeta) {
    return {
      result,
      meta: {
        decision_source: decisionSource
      }
    }
  }

  return result
}

export async function evaluateGreetings(transcriptPayload, options = {}) {
  const segments = extractTranscriptSegments(transcriptPayload)

  if (segments.length === 0) {
    return withMeta(
      buildGreetingOutput({
      detected: false,
      reason:
        'No timestamped transcript segments were detected, so greeting cannot be validated in the first 20% window.'
      }),
      'none',
      options
    )
  }

  const totalDuration = getTotalCallDurationSeconds(segments)
  const detectionWindowSeconds = totalDuration * 0.2

  const csmWindowSegments = segments.filter(
    (segment) => isCsmSpeaker(segment.speaker) && segment.timestampSeconds <= detectionWindowSeconds
  )

  if (csmWindowSegments.length === 0) {
    return withMeta(
      buildGreetingOutput({
      detected: false,
      reason:
        'No CSM segment was found within the first 20% of the call duration, so greeting is marked as not detected.'
      }),
      'none',
      options
    )
  }

  const greetingPhrases = await loadDictionaryPhrases([ 'greetings_output.json'])
  const repositoryOrFlexible = matchSegmentsAgainstPhrases(csmWindowSegments, greetingPhrases)

  if (repositoryOrFlexible.matched) {
    return withMeta(
      buildGreetingOutput({
      detected: true,
      matchedPhrase: repositoryOrFlexible.matchedPhrase,
      timestamp: repositoryOrFlexible.timestamp,
      reason:
        repositoryOrFlexible.mode === 'repository'
          ? 'CSM greeted the client within the first 20% of the call duration (repository match).'
          : 'CSM greeted the client within the first 20% of the call duration (flexible phrase match).'
      }),
      'repository',
      options
    )
  }

  const aiFallback = await runOpenAiIntentFallback({
    parameterName: 'Greeting Detection',
    instructions:
      'Decide whether the CSM delivered a valid greeting in these CSM segments from the first 20% of the call. Mark detected true only when greeting intent is clearly present.',
    segments: csmWindowSegments
  })

  if (aiFallback?.detected) {
    return withMeta(
      buildGreetingOutput({
      detected: true,
      matchedPhrase: aiFallback.text,
      timestamp: aiFallback.timestamp,
      reason:
        aiFallback.reason ||
        'CSM greeted the client within the first 20% of the call duration (OpenAI fallback).'
      }),
      'openai',
      options
    )
  }

  return withMeta(
    buildGreetingOutput({
      detected: false,
      reason: 'No greeting by the CSM was detected within the first 20% of the call duration.'
    }),
    'none',
    options
  )
}
