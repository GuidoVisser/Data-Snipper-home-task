using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FinerDistilBert
{
    /**
     * Utility functions for the project
     */
    internal class ProjectUtils
    { 
        /**
         * Checks if the content of a string is a number
        */
        static public bool StringIsNumber(string word)
        {
            float n;
            return float.TryParse(word, out n);
        }

        /**
         * Gets the index of the largest value in a given span
         */
        static public int GetMaxValueIndex(ReadOnlySpan<float> span)
        {
            float maxVal = span[0];
            int maxIndex = 0;
            for (int i = 1; i < span.Length; ++i)
            {
                var v = span[i];
                if (v > maxVal)
                {
                    maxVal = v;
                    maxIndex = i;
                }
            }
            return maxIndex;
        }
    }
}
