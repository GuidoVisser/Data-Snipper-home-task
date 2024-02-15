using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Linq;
using Excel = Microsoft.Office.Interop.Excel;
using Office = Microsoft.Office.Core;
using Microsoft.Office.Tools.Excel;
using OnnxRuntime = Microsoft.ML.OnnxRuntime;
using Microsoft.Office.Interop.Excel;

using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

// TODOs
// [ ] Tokenize the input
// [ ] Load model
// [ ] Create Tensor based on input
// [ ] Do model inference
// [ ] Show model output

namespace FinerDistilBert_VSTO
{
    public struct BertInput
    {
        public long[] InputIds { get; set; }
        public long[] AttentionMask { get; set; }
        //public long[] TypeIds { get; set; }
    }

    public partial class ThisAddIn
    {
        private void ThisAddIn_Startup(object sender, System.EventArgs e)
        {
            this.Application.WorkbookBeforeSave += new Microsoft.Office.Interop.Excel.AppEvents_WorkbookBeforeSaveEventHandler(Application_WorkbookBeforeSave);
            this.Application.WorkbookNewSheet += new AppEvents_WorkbookNewSheetEventHandler(AddUserPrompts);
        }

        private void ThisAddIn_Shutdown(object sender, System.EventArgs e)
        {
        }

        private string GetDummieInput()
        {
            return "Here is a dummie string";
        }

        private Excel.Worksheet GetActiveWorksheet()
        {
            return ((Excel.Worksheet)Application.ActiveSheet);
        }

        private string GetUserInput()
        {
            
            return GetActiveWorksheet().get_Range("B1").Value2;
        }

        void AddUserPrompts(Microsoft.Office.Interop.Excel.Workbook wb, object Sh)
        {
            Excel.Range a1 = ((Excel.Worksheet)Sh).get_Range("A1");
            a1.Value2 = "Add input here: ";

        }

        void ShowTokenizedInput()
        {
            Excel.Worksheet activeWorksheet = GetActiveWorksheet();
            Excel.Range thirdRow = activeWorksheet.get_Range("A3");
            thirdRow.EntireRow.Insert(Excel.XlInsertShiftDirection.xlShiftDown);
            Excel.Range newThirdRow = activeWorksheet.get_Range("A3");
            newThirdRow.Value2 = GetUserInput();

            var pathtomodel = "C:/Users/Guido/Projects/datasnipper_hometask/models/onnx/DataSnipper_FinerDistilBert.onnx";
            //var runOptions = new RunOptions();
            //var session = new InferenceSession(pathtomodel);

            int x = 1;
        }

        void Application_WorkbookBeforeSave(Microsoft.Office.Interop.Excel.Workbook Wb, bool SaveAsUI, ref bool Cancel)
        {
            ShowTokenizedInput();
        }

        #region VSTO generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InternalStartup()
        {
            this.Startup += new System.EventHandler(ThisAddIn_Startup);
            this.Shutdown += new System.EventHandler(ThisAddIn_Shutdown);
        }
        
        #endregion
    }
}
