using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Collections.Specialized.BitVector32;

namespace FinerDistilBert
{
    /**
     * Provides functionality for Inference with a pretrained ONNX model. when instantiated will ask user for a path to a trained model.
     */
    internal class FinerDistilBert_Model
    {
        public FinerDistilBert_Model()
        {
            runOptions = new RunOptions();
            inferenceSession = GetInferenceSession();
        }

        public IDisposableReadOnlyCollection<OrtValue> DoInference(IReadOnlyDictionary<string, OrtValue> input)
        {
            return inferenceSession.Run(runOptions, input, inferenceSession.OutputNames);
        }

        string GetDefaultModelLocation()
        {
            var currentPath = System.IO.Directory.GetCurrentDirectory();
            var modelPath = "\\..\\..\\..\\..\\models\\onnx\\DataSnipper_FinerDistilBert.onnx";

            var combined = currentPath + modelPath;

            return Path.GetFullPath(combined);
        }

        InferenceSession GetInferenceSession()
        {
            Console.WriteLine("Please provide a path to an ONNX model. Leave blank for default (" + GetDefaultModelLocation() + "):");
            var modelPath = Console.ReadLine();
            if (modelPath == "")
            {
                modelPath = GetDefaultModelLocation();
            }

            try
            {
                return new InferenceSession(modelPath);
            }
            catch (Exception e)
            {
                Console.WriteLine("Invalid model path. Try again.");
                return GetInferenceSession();
            }
        }

        public RunOptions runOptions;
        public InferenceSession inferenceSession;
    }
}
