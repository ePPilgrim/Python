using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TFService
{
//    public class ModernHistoricalSimulationLogicalForm : ILogicalForm, IDisposable
//    {

//        private double iter = 0.0;

//        private readonly ApplicationBinding<ModernHistoricalSimulation> Binding;

//        public CommandElement Fetch { get; set; }
//        public CommandElement Run { get; set; }
//        public TypedElement<double> KLD { get; set; }
//        public ListView<DistributionGrid> Grid { get; set; }

//        public ModernHistoricalSimulationLogicalForm(ILogicalViewController controller, IViewElementFactory factory, IAgent agent)
//        {
//            Binding = ApplicationBinding<ModernHistoricalSimulation>.CreateAndInitialize(controller, agent);

//            Grid = factory.NewGridElement(controller.CreateControlledView<DistributionGrid>(nameof(Grid)))
//            .WithSelectedColumns(new[] {
//                    nameof(DistributionGrid.Original),
//                    nameof(DistributionGrid.Approximation),
//                    nameof(DistributionGrid.Deviation)
//            }).Build();

//            Fetch = factory.NewCommandElement().WithTitle("Fetch Data").Build();
//            Fetch.DoExecute += OnFetchData;

//            Run = factory.NewCommandElement().WithTitle("Sent Request").Build();

//            KLD = factory.NewTypedElement<double>().WithMaxLength(10).WithTitle("KLD distance").Build();

//            controller.InitializeView(this);
//        }

//        private void OnFetchData(object sender, CommandEventArgs e)
//        {
//            if (Grid.Items.Any()) { Grid.DeleteItems(Grid.Items.Select(item => item.Id).ToList()); }
//            var data = new[] {
//                    new {Original = 0.1, Approximated = 0.2},
//                    new {Original = 0.3, Approximated = 0.4},
//                    new {Original = 0.5, Approximated = 0.6}
//                }.Select(x => new { Original = x.Original + iter, Approximated = x.Approximated + iter }).ToList();
//            Grid.AddItems(data);
//            iter += 1.0;
//        }
            
//            public void Dispose()
//        {
//            //throw new NotImplementedException();
//        }

//        public TypedElement<double> Original { get; }
//        public TypedElement<double> Approximation { get; }
//        public TypedElement<double> Deviation { get; }⌈
//        }
//}
}
