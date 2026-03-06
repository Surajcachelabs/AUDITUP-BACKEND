import 'dotenv/config'
import express from 'express'
import cors from 'cors'
import { evaluateGreetings } from './src/greetings.js'
import { evaluateClosingStatement } from './src/closingStatement.js'
import { evaluateSummery } from './src/summery.js'
import { evaluateLegalDisclamer } from './src/LegalDisclamer.js'

const app = express()
const port = Number(process.env.PORT || 8000)

app.use(cors())
app.use(express.json({ limit: '5mb' }))

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'audit-demo-backend'
  })
})

app.post('/api/evaluate', async (req, res) => {
  try {
    const { transcript } = req.body ?? {}

    if (!transcript) {
      return res.status(400).json({
        error: 'Missing transcript in request body'
      })
    }

    const greetings = await evaluateGreetings(transcript)
    const closingStatement = await evaluateClosingStatement(transcript)
    const summery = await evaluateSummery(transcript)
    const legalDisclamer = await evaluateLegalDisclamer(transcript)

    const totalScore =
      greetings.score + closingStatement.score + summery.score + legalDisclamer.score
    const maxScore = 4

    return res.json({
      score_summary: {
        total_score: totalScore,
        max_score: maxScore,
        percentage: Number(((totalScore / maxScore) * 100).toFixed(2))
      },
      results: {
        greetings,
        closingStatement,
        summery,
        legalDisclamer
      },
      generated_at_utc: new Date().toISOString()
    })
  } catch (error) {
    return res.status(500).json({
      error: 'Failed to evaluate transcript',
      details: error instanceof Error ? error.message : 'Unknown error'
    })
  }
})

app.listen(port, () => {
  console.log(`Backend running on http://localhost:${port}`)
})
