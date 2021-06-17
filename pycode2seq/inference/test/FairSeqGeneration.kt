package ai.grazie.nlc.local.generation

import ai.grazie.nlc.local.CompletionConfig
import ai.grazie.nlc.local.generation.matcher.FuzzyPrefixMatcher
import ai.grazie.nlc.local.generation.matcher.PrefixMatcher
import ai.grazie.nlc.local.generation.model.GenerationModel
import ai.grazie.nlc.local.generation.search.BeamSearch
import ai.grazie.nlc.local.generation.search.Search
import ai.grazie.nlc.local.tokenizer.BPETokenizer
import kotlin.math.*

internal class FairSeqGeneration(private val model: GenerationModel, private val tokenizer: BPETokenizer) {
    data class PrefixInfo(val text: String, val errLimit: Int)

    private val prefixMatcher = FuzzyPrefixMatcher(tokenizer)

    private var contexts: Array<MutableList<Int>>? = null
    private var generationState: GenerationModel.State? = null
    private var prefixes: List<PrefixInfo>? = null
    private var eachStepProbs: List<MutableList<Double>> = listOf(ArrayList())
    private var nextLogProbs: Array<DoubleArray>? = null

    private val vocabSize: Int
        get() = tokenizer.vocabSize

    private var logSpellProb = ln(0.0001)

    private fun getSearch(config: CompletionConfig.Generation): Search {
        require(config.numGroups == 1) { "num groups > 1 is not supported" }

        return BeamSearch(vocabSize, config.numBeams, config.repetitionPenalty)
    }

    private fun modifyScore(scores: Array<DoubleArray>): Array<DoubleArray> {
        prefixes!!.forEachIndexed { i, (prefix, err_limit) ->
            if (prefix.isEmpty()) return@forEachIndexed

            val prefixIndsByErr = prefixMatcher.prefixTokensByErr(prefix, err_limit)
            for (j in prefixIndsByErr.notMatchedTokens) {
                scores[i][j] = Double.NEGATIVE_INFINITY
            }

            prefixIndsByErr.tokensByErrCount.forEachIndexed { errNum, prefixToken ->
                if (errNum != 0) {
                    for (j in prefixToken) {
                        scores[i][j] = scores[i][j] + (errNum + 1) * logSpellProb
                    }
                }
            }

            // ban tokens with bad symbols
//            for (j in tokenizer.invalidIds) {
//                scores[i][j] = Double.NEGATIVE_INFINITY
//            }
        }

        return scores
    }

    private fun initState(context: IntArray, prefix: String, config: CompletionConfig.Generation) {
        contexts = arrayOf(context.asList().toMutableList())
        generationState = model.createState()
        logSpellProb = ln(config.spellProb)
        eachStepProbs = listOf(ArrayList())
        prefixes = listOf(PrefixInfo(prefix, config.prefixErrLimit))
    }

    private fun sortState(sortMask: IntArray) {
//        contexts = contexts!!.slice(sortMask)
        contexts = sortMask.map { ArrayList(contexts!![it]) }.toTypedArray()
        generationState!!.update(sortMask)
        eachStepProbs = sortMask.map { ArrayList(eachStepProbs[it]) }
        prefixes = prefixes!!.slice(sortMask)
    }

    private fun updatePrefix(newTokensIds: IntArray) {
        if (prefixes == null) return

        val result = ArrayList<PrefixInfo>(prefixes!!.size)

        prefixes!!.forEachIndexed { i, (prefix, errLimit) ->
            val tokenId = newTokensIds[i]
            val token = tokenizer.decode(tokenId)
            val prefixLen = min(prefix.length, token.length)
            val errCnt = PrefixMatcher.levenshtein(prefix.substring(0, prefixLen), token.substring(0, prefixLen))
            val newPrefix = prefix.substring(prefixLen)
            result.add(PrefixInfo(newPrefix, min(errLimit - errCnt, newPrefix.length)))
        }

        prefixes = result
    }

    private fun updateState(sortMask: IntArray, newTokensIds: IntArray) {
        sortState(sortMask)

        sortMask.zip(newTokensIds).forEachIndexed {
            index, (batchInd, tokenInd) ->
            run {
                eachStepProbs[index].add(exp(nextLogProbs!![batchInd][tokenInd]))
                contexts!![index].add(tokenInd)
            }
        }

        updatePrefix(newTokensIds)
        val indexOfMatchedSeqs = prefixes!!.mapIndexed { i, p -> Pair(i, p) }.filter { pair -> pair.second.errLimit >= 0 }.map { it.first }
        sortState(indexOfMatchedSeqs.toIntArray())
    }

    private fun updateScores() {
        val probs = model.nextProbs(contexts!!, generationState!!)
        val logProbs = log(probs)

        nextLogProbs = modifyScore(logProbs)
    }

    private fun currentHypothesis(search: Search): List<GenerationInfo> {
        return search.hypotheses().zip(eachStepProbs).map { (hyp, probs) -> GenerationInfo(probs, hyp) }
    }

    fun generate(context: IntArray, prefix: String, config: CompletionConfig.Generation): List<List<GenerationInfo>> {
        val search = getSearch(config)

        initState(context, prefix, config)
        updateScores()
//        sortState(IntArray(search.batchSize))

        val result = ArrayList<List<GenerationInfo>>()
        for (i in 0 until config.maxLen) {
            val stepResult = search.step(nextLogProbs!!, context)
            updateState(stepResult.sortMask, stepResult.newTokens)

            if (i < config.maxLen - 1) {
                updateScores()
            }
            result.add(currentHypothesis(search))

            if (contexts!!.isEmpty())
                break
        }

        return result
    }
}
