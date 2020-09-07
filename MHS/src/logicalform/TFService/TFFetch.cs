using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Threading.Tasks;

namespace TFService
{
     public class TFService {
        private readonly string PathToData;
        private readonly string UrlToTf;
        private readonly int MaxLine;
        private const int NumberOfLine = 3000;
        public TfRequest Observables = new TfRequest();
        public IList<double> Target = new List<double>();

        public TFService(string pathToData, string urlToTf, int maxLine) {
            PathToData = pathToData;
            UrlToTf = urlToTf;// "http://localhost:8501/v1/models/my_model:predict";
            MaxLine = maxLine;
            Target = new List<double>(new double[NumberOfLine]);
            Observables.instances = new List<IList<float>>(NumberOfLine);
            var sz = 21 + 21 + 12;
            for (int i = 0; i < NumberOfLine; ++i) {
                Observables.instances.Add(new List<float>(new float[sz]));

            }
        }

        public IList<double> GetPostData() {
            var httpWebRequest = (HttpWebRequest)WebRequest.Create(UrlToTf);
            httpWebRequest.ContentType = "application/json";
            httpWebRequest.Method = "POST";

            using (var streamWriter = new StreamWriter(httpWebRequest.GetRequestStream())) {
                string json = JsonConvert.SerializeObject(Observables);
                streamWriter.Write(json);
            }

            var httpResponse = (HttpWebResponse)httpWebRequest.GetResponse();
            string res = null;
            using (var streamReader = new StreamReader(httpResponse.GetResponseStream())) {
                res = streamReader.ReadToEnd();
            }

            return JsonConvert.DeserializeObject<TfResponse>(res).predictions.Select(x => x.FirstOrDefault()).ToList();
        }

        public int FetchData() {
            for (int i = 0; i < Observables.instances.Count; ++i) {
                for (int j = 0; j < Observables.instances[i].Count; ++j) Observables.instances[i][j] = 0.0f;
            }
            var rnd = new Random();
            var fromLine = rnd.Next(1, MaxLine - NumberOfLine);
            using (var fs = new StreamReader(PathToData)) {
                for (int i = 0; i < fromLine + NumberOfLine; ++i) {
                    var item = fs.ReadLine();
                    if (i < fromLine) continue;
                    var strv = item.Split(',');
                    assignObservable(strv, i - fromLine);
                    assignTarget(strv, i - fromLine);
                }
            }
            return fromLine;
        }

        void assignTarget(string[] strv, int v) {
            var cnt = strv.Length - 3;
            var sum = 0.0;
            for (int i = 0; i < 12; ++i) {
                sum += convertToDouble(strv[cnt - i]);
            }
            Target[v] = sum;
        }

        void assignObservable(string[] strv, int v) {
            var ycv = Enumerable.Range(3, 42).Select(x => (float)convertToDouble(strv[x])).ToList();
            var freq = Convert.ToInt32(strv[strv.Length - 1]);
            var ifreq = Enumerable.Range(1, 12).Select(x => x * freq).Where(x => x <= 12).Select(x => x + 41);
            for (int i = 0; i < 42; ++i) {
                Observables.instances[v][i] = ycv[i];
            }
            foreach (var i in ifreq) Observables.instances[v][i] = 1.0f;
        }

        double convertToDouble(string str) => Convert.ToDouble(str, System.Globalization.CultureInfo.InvariantCulture);

        public IList<CashFlows> MergeToGrid(IList<double> actual, IList<double> predicted = null) {

            if (predicted == null) {
                return Enumerable.Range(0, NumberOfLine).Select(x => new CashFlows(){ True = actual[x], Predicted = 0.0, AbsRelError = 100.0 }).ToList();
            }
            return Enumerable.Range(0, NumberOfLine).Select(x => new CashFlows(){
                True = actual[x], Predicted = predicted[x],
                AbsRelError = Math.Abs(100.0 * ((actual[x] - predicted[x])/actual[x]))
            }).ToList();
        }

        public IList<VaR> VarToGrid(List<double> actual, List<double> predicted) {
            double[] risks = new double[] { 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 };
            string[] repRisks = new string[] { "99%", "98%", "97%", "96%", "95%", "94%", "93%", "92%", "91%", "90%" };

            actual.Sort();
            predicted.Sort();

            return Enumerable.Range(0, risks.Length)
                        .Select(x => new Tuple<string, int>(repRisks[x], Convert.ToInt32(risks[x] * NumberOfLine)))
                        .Select(x => new VaR(){ Risk = x.Item1, TrueVaR = actual[x.Item2], PredictedVaR = predicted[x.Item2] }).ToList();
        }

    }
}
