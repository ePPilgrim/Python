using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TFService
{

    class Program
    {
        static void Main(string[] args)
        {
            var num = Convert.ToDouble("0.003341", System.Globalization.CultureInfo.InvariantCulture);
            var obj = new TFService(@"C:\Users\PLDD\Practice\GitHub\Python\MHS\data\train\train_real_0_0_410_38_40938.txt", "http://localhost:8501/v1/models/my_model:predict", 40930);
            obj.FetchData();
            var predicted = obj.GetPostData();
            var cashflows = obj.MergeToGrid(obj.Target, predicted);
            var vars = obj.VarToGrid(cashflows);
            foreach(var val in vars)
            {
                Console.WriteLine($"{val.Risk} -- {val.TrueVar} -- {val.PredictedVar}");
            }

        }
    }
}
