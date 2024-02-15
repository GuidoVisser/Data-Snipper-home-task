using BERTTokenizers;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinerDistilBert
{
    internal class FinerDistilBert_ModelInput
    {
        /**
         * Tokenizes the input text and provides functionality to allign the encoded tokens and the words in the input text. Furthermore can be used to get the ONNX input for inference with an Onnx Runtime session.
         */
        public FinerDistilBert_ModelInput(string inputText)
        {
            rawInput = inputText;

            // Get the sentence tokens.
            tokens = tokenizer.Tokenize(inputText);

            inputWords = SplitIntoWords(inputText);
            tokenWordIndices = GetTokenWordMapping(ref tokenWordIndices, ref inputWords, ref tokens, 0, 0, "", true);

            int x = 1;
        }

        bool CharIsPartOfNumber(int i, string inputText) 
        {   
            if (System.Char.IsDigit(inputText[i - 1]) && System.Char.IsDigit(inputText[i + 1])) return true;
            return false;
        }

        private string[] SplitIntoWords(string inputText)
        {
            string specialChars = "~`!@#$€£%^&*()_+=-[]{};:\'\"\\|/?.,<>"; // How do I add other valuta chars?

            List<int> insertAt = new List<int>();
            for (int i = 0; i<inputText.Length; i++)
            {
                if (i <= 1 || i == inputText.Length - 1) continue;
                if (CharIsPartOfNumber(i, inputText)) continue;
                if (specialChars.Contains(inputText[i]))
                {
                    insertAt.Add(i);
                }
            }

            insertAt.Reverse();
            foreach (int i in insertAt) {
                if (inputText[i+1] != ' ') {
                    inputText = inputText.Insert(i + 1, " ");
                }

                if (inputText[i-1] != ' ') {
                    inputText = inputText.Insert(i, " ");
                }
            }

            return inputText.ToLower().Split(" ");
        }

        public IReadOnlyDictionary<string, OrtValue> GetONNXInput()
        {

            // Encode the sentence and pass in the count of the tokens in the sentence.
            var encoded = tokenizer.Encode(tokens.Count(), rawInput);

            // Break out encoding to InputIds, AttentionMask and TypeIds from list of (input_id, attention_mask, type_id).
            var bertInput = new BertInput()
            {
                InputIds = encoded.Select(t => t.InputIds).ToArray(),
                AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
            };

            // Create input tensors over the input data.
            var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(
                bertInput.InputIds,
                new long[] { 1, bertInput.InputIds.Length }
            );

            var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(
                bertInput.AttentionMask,
                new long[] { 1, bertInput.AttentionMask.Length }
            );

            // Create input data for session. Request all outputs in this case.
            var inputs = new Dictionary<string, OrtValue>
            {
                { "input_ids", inputIdsOrtValue },
                { "attention_mask", attMaskOrtValue },
            };

            return inputs;
        }

        public bool TokenIsValidStartOfWord(int tokenIndex)
        {
            if (TokenIsStartOfWord(tokenIndex) && tokenWordIndices[tokenIndex].wordIdx != -1) return true;
            return false;
        }

        public bool TokenIsStartOfWord(int tokenIndex)
        {
            return tokenWordIndices[tokenIndex].isStartOfWord;
        }

        public string GetWord(int tokenIndex) 
        {
            return inputWords[tokenWordIndices[tokenIndex].wordIdx];
        }

        public string GetSegmentOfInput(Range range)
        {
            return string.Join(" ", inputWords[range]);
        }

        public int GetWordIndex(int tokenIndex)
        {
            return tokenWordIndices[tokenIndex].wordIdx;
        }

        public int GetNrWords()
        { 
            return inputWords.Length; 
        }

        static private bool IsSpecialToken(string token)
        {
            string[] specialTokens = ["[CLS]", "[MASK]", "PAD", "[SEP]", "[UNK]"];
            return specialTokens.Contains(token);
        }

        static private bool SpecialTokenIsAdded(string token)
        {
            string[] yes = ["[CLS]", "[PAD]", "[SEP]"];
            return yes.Contains(token);
        }

        static private string GetTokenAddValue(string token)
        {
            if (token.StartsWith("##"))
            {
                return token.TrimStart('#');
            }
            else
            {
                return token;
            }
        }
        private List<TokenWordIndex> GetTokenWordMapping(ref List<TokenWordIndex> alreadyDone, ref string[] sentenceWords, ref List<(string Token, int, long)> tokenList, int curWordIndex, int curTokenIndex, string workingToken, bool isNewWord)
        {
            // Done with recursion
            if (curTokenIndex == tokenList.Count) return alreadyDone;

            TokenWordIndex newEntry = new TokenWordIndex(curWordIndex, isNewWord, true);

            isNewWord = true;
            int incrementWordIndex = 1;
            if (IsSpecialToken(tokenList[curTokenIndex].Token)) // Is special token, ignore as word
            {
                newEntry.tokenIsWord = false;
                newEntry.wordIdx = -1;
                incrementWordIndex = 0;
                if (!SpecialTokenIsAdded(tokenList[curTokenIndex].Token))
                {
                    incrementWordIndex = 1;
                }
                workingToken = "";
            }
            else if (tokenList[curTokenIndex].Token == sentenceWords[curWordIndex])
            { // singular token is next word
                workingToken = "";
            }
            else if (workingToken + GetTokenAddValue(tokenList[curTokenIndex].Token) == sentenceWords[curWordIndex])
            { // current token added to workingToken is entire word
                workingToken = "";
            }
            else
            {    // token is only part of the current word
                incrementWordIndex = 0;
                workingToken = workingToken + GetTokenAddValue(tokenList[curTokenIndex].Token);
                isNewWord = false;
            }

            // Add entry and continue recursion
            alreadyDone.Add(newEntry);
            curWordIndex = curWordIndex + incrementWordIndex;
            return GetTokenWordMapping(ref alreadyDone, ref sentenceWords, ref tokenList, curWordIndex, ++curTokenIndex, workingToken, isNewWord);
        }


        private BertUncasedBaseTokenizer tokenizer = new BertUncasedBaseTokenizer();
        private List<TokenWordIndex> tokenWordIndices = new List<TokenWordIndex>();
        public string[] inputWords;

        string rawInput;
        private List<(string Tokens, int, long)> tokens;
    }
}
