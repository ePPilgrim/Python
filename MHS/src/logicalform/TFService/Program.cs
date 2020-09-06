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
            var obj = new TFService(@"C:\Users\PLDD\Practice\GitHub\Python\MHS\data\train\repository\test_102229", "http://localhost:8501/v1/models/my_model:predict", 102000);
            obj.FetchData();
            var obs = obj.Observables.instances;
            //foreach(var val1 in obs)
            //{
            //    foreach(var val2 in val1)
            //    {
            //        Console.Write($"{val2}, ");
            //    }
            //    Console.WriteLine("");
            //}

            //foreach(var val in obj.Target)
            //{
            //    Console.Write($"{val}, ");
            //}
            //Console.WriteLine("");
            var predicted = obj.GetPostData().ToList();
            //foreach (var val in predicted)
            //{
            //    Console.Write($"{val}, ");
            //}
            //Console.WriteLine("");
            //Console.WriteLine(obj.jjson);
            var rnd = new Random();
            
            var cashflows = obj.MergeToGrid(obj.Target, predicted);

            int i = 0;
            foreach(var val in cashflows)
            {
                var pos = rnd.Next(0, 3000);
                if( pos % 2 == 0)
                {
                    Console.WriteLine($"{val}");
                    i++;
                }
                if(i > 10)
                {
                    break;
                }
            }
            var vars = obj.VarToGrid(obj.Target.ToList(), predicted);
            foreach(var val in vars)
            {
                Console.WriteLine($"{val}");
            }

        }
    }
}
