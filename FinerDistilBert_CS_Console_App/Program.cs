using System;

using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using BERTTokenizers;
using BERTTokenizers.Base;
using System.Text;

namespace FinerDistilBert
{
    internal class FinerDistilBertInference
    {
        /**
         * Main loop of the program. Consists of asking the user for a path to an Onnx model and an input text and proceeds to tag the numbers in the input text using the pretrained model.
         */
        static void Main(string[] args)
        {
            /**
            * Gets an input text segment from the user. A default input segment is provided with a good number of numbers of the negative and positive Finer-139 classes.
            */
            string GetInput()
            {   // €
                //string testInput = "For both the three and six months ended May 31 , 2016 , 312 homesites were sold to Lennar by one of the Company 's unconsolidated entities for $ 92.0 million that resulted in $ 29.7 million , of gross profit , of which the Company 's portion was deferred . For the three months ended May 31 , 2015 , Lennar Homebuilding equity in earnings included $ 11.6 million of equity in earnings from one of the Company 's unconsolidated entities primarily due to sales of approximately 60 homesites and a commercial property to third parties for $ 121.3 million that resulted in $ 37.6 million of gross profit . For the six months ended May 31 , 2015 , Lennar Homebuilding equity in earnings included $ 43.0 million of equity in earnings from one of the Company 's unconsolidated entities primarily due to ( 1 ) sales of approximately 660 homesites to third parties for $ 407.2 million that resulted in $ 138.4 million of gross profit and ( 2 ) sales of 300 homesites to Lennar for $ 126.4 million that resulted in $ 44.6 million of gross profit , of which the Company 's portion was deferred . Balance Sheets On May 2 , 2016 ( the “ Closing Date ” ) , the Company contributed , or obtained the right to contribute , its investment in three strategic joint ventures previously managed by Five Point Communities in exchange for an investment in a newly formed Five Point entity .";
                string testInput = "For both the three and six months ended May 31, 2016, 312 homesites were sold to Lennar by one of the Company's unconsolidated entities for $ 92.0 million that resulted in $29.7 million, of gross profit, of which the Company's portion was deferred. For the three months ended May 31, 2015, Lennar Homebuilding equity in earnings included $ 11.6 million of equity in earnings from one of the Company's unconsolidated entities primarily due to sales of approximately 60 homesites and a commercial property to third parties for $ 121.3 million that resulted in $ 37.6 million of gross profit. For the six months ended May 31, 2015, Lennar Homebuilding equity in earnings included $ 43.0 million of equity in earnings from one of the Company's unconsolidated entities primarily due to (1) sales of approximately 660 homesites to third parties for $ 407.2 million that resulted in $ 138.4 million of gross profit and (2) sales of 300 homesites to Lennar for $ 126.4 million that resulted in $ 44.6 million of gross profit, of which the Company's portion was deferred. Balance Sheets On May 2, 2016 (the \"Closing Date” ), the Company contributed, or obtained the right to contribute, its investment in three strategic joint ventures previously managed by Five Point Communities in exchange for an investment in a newly formed Five Point entity.";

                // Get user input
                Console.WriteLine("Please enter a piece of text you wish to process. Leave blank for default");
                var sentence = Console.ReadLine();
                if (sentence == null || sentence.Length <= 1)
                {
                    sentence = testInput;
                }

                // BERTTokenizer breaks on these characters..
                sentence = sentence.Replace("€", "$"); // € character not present in train set, so it messes up the context (apparently quite important for good results..)
                sentence = sentence.Replace("£", "$"); // idem
                sentence = sentence.Replace("”", "\"");
                sentence = sentence.Replace("“", "\"");

                Console.WriteLine(sentence);
                return sentence;
            }

            var classNames = new ClassNames();

            FinerDistilBert_Model model = new FinerDistilBert_Model();

            string userInput = GetInput();
            var modelInput = new FinerDistilBert_ModelInput(userInput);


            //using var output = model.inferenceSession.Run(model.runOptions, modelInput.GetONNXInput(), model.inferenceSession.OutputNames);
            using var output = model.DoInference(modelInput.GetONNXInput());

            var outputInfo = output[0].GetTensorTypeAndShape();
            var outputArray = output[0].GetTensorDataAsSpan<float>();
            int logitLength = (int)outputInfo.Shape[2];

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.Write("######################################\n\n");
            Console.ResetColor();
            for ( int i = 0; i < outputInfo.Shape[1]; i++ )
            {
                if (!modelInput.TokenIsValidStartOfWord(i)) continue;

                var tokenOutput = outputArray.Slice(i * logitLength, logitLength);

                string word = modelInput.GetWord(i);
                int wordIndex = modelInput.GetWordIndex(i);
                
                if (ProjectUtils.StringIsNumber(word))
                {
                    string before = "";
                    if (wordIndex != 0) {
                        int beforeSize = Math.Min(10, wordIndex);
                        int start = wordIndex - beforeSize;
                        int end = wordIndex;

                        string prefix = "";
                        if (start != 0)
                        {
                            prefix = "...";
                        }

                        before = prefix + modelInput.GetSegmentOfInput(start..end);
                    }

                    string after = "";
                    if (wordIndex != modelInput.GetNrWords())
                    {
                        int afterSize = Math.Min(10, modelInput.GetNrWords() - wordIndex);
                        int start = wordIndex + 1;
                        int end = wordIndex + afterSize;

                        string suffix = "";
                        if (end != modelInput.GetNrWords()) 
                        { 
                            suffix = "..."; 
                        }

                        after = modelInput.GetSegmentOfInput(start..end) + suffix;
                    }

                    Console.WriteLine("Class: " + classNames.GetName((ProjectUtils.GetMaxValueIndex(tokenOutput))));
                    Console.WriteLine("");
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.Write(before);
                    Console.ResetColor();
                    Console.Write(" >" + word + "< ");
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.Write(after);
                    Console.Write("\n\n######################################\n\n");
                    Console.ResetColor();
                }
            }
            Console.ReadKey(true);
        }
    }

    public struct BertInput
    {
        public long[] InputIds { get; set; }
        public long[] AttentionMask { get; set; }
    }

    public struct TokenWordIndex
    {
        public TokenWordIndex(int wordidx, bool isstartofword, bool tokenisword)
        {
            wordIdx = wordidx;
            isStartOfWord = isstartofword;
            tokenIsWord = tokenisword;
        }

        public int wordIdx { get; set; }
        public bool isStartOfWord { get; set; }
        public bool tokenIsWord { get; set; }
    }
}