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
        public double True;
        public double Predicted;
        public double AbsRelError;

        public string ToString()
        {
            return $"True = {True}, Predicted = {Predicted}, AbsRelError = {AbsRelError}";
        }
    }

    public class VaR
    {

        public string Risk;
        public double TrueVaR;
        public double PredictedVaR;
        public string ToString()
        {
            return $"Risk = {Risk}, TrueVaR = {TrueVaR}, PredictedVaR = {PredictedVaR}";
        }
    }
}
