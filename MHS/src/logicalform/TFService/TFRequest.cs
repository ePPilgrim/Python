using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TFService
{
    public struct TfRequest
    {
        public string signature_name => "serving_default";
        public IList<IList<float>> instances;
    }

    public struct TfResponse {
        public IList<IList<double>> predictions;
    }

    public class CashFlows
    {
        public CashFlows(double aTrue = 0.0, double aPredicted = 0.0)
        {
            True = aTrue;
            Predicted = aPredicted;
        }
        public double True;
        public double Predicted;
    }

    public class VaR
    {
        public VaR(string aRisk, double aTrueVar, double aPredictedVar)
        {
            Risk = aRisk;
            TrueVar = aTrueVar;
            PredictedVar = aPredictedVar;
        }

        public VaR(double aTrueVar, double aPredictedVar)
        {
            Risk = "";
            TrueVar = aTrueVar;
            PredictedVar = aPredictedVar;
        }

        public string Risk;
        public double TrueVar;
        public double PredictedVar;
    }
}
