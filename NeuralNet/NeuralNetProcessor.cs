using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.IO;
using Layer = System.Collections.Generic.List<NeuralNet.Neuron>;
namespace NeuralNet
{
    class NeuralNetProcessor
    {
        static void Main(string[] args)
        {

            //CreateTrainingFile("test");
            
            //need to include iterations
            TrainingData trainData = new TrainingData("train.txt");
            TrainingData testData = new TrainingData("test.txt");

            // e.g., { 3, 2, 1 }
            List<int> topology = new List<int>();
            List<double> inputVals = new List<double>();
            List<double> targetVals = new List<double>();
            List<double> resultVals = new List<double>();
            trainData.getTopology(topology);

            NeuralNet myNet = new NeuralNet(topology);

            int trainingPass = 0;
            while (!trainData.isEof()) {
                ++trainingPass;
                Console.Write("\nPass " + trainingPass);
                // Get new input data and feed it forward:
                if (trainData.getNextInputs(inputVals) != topology[0]) {
                    break;
                }
                showVectorVals("Inputs:", inputVals);
                myNet.feedForward(inputVals);
                // Collect the net's actual output results:
                myNet.getResults(resultVals);
                showVectorVals("Outputs:", resultVals);
                // Train the net what the outputs should have been:
                trainData.getTargetOutputs(targetVals);
                showVectorVals("Targets:", targetVals);
                Debug.Assert(targetVals.Count == topology.Last());
                myNet.backProp(targetVals);
                // Report how well the training is working, average over recent
                // samples:
                Console.WriteLine("Net recent average error: " 
                    + myNet.RecentAverageError);
                //if ((trainingPass > 100) && (myNet.RecentAverageError < .03)) break; 
            }
            trainData.close();

            testData.getTopology(topology);

            while (!testData.isEof())
            {
                Console.WriteLine("test pass");
                if (testData.getNextInputs(inputVals) != topology[0])
                {
                    break;
                }
                //test data
                showVectorVals("Inputs:", inputVals);
                myNet.feedForward(inputVals);
                // Collect the net's actual output results:
                myNet.getResults(resultVals);
                showVectorVals("Outputs:", resultVals);
                // Train the net what the outputs should have been:
                testData.getTargetOutputs(targetVals);
                showVectorVals("Targets:", targetVals);
                Debug.Assert(targetVals.Count == topology.Last());
            }
            testData.close();

            Console.WriteLine("\nDone");



            /*
            List<int> topology = new List<int>();
            List<double> inputVals = new List<double>();
            List<double> targetVals = new List<double>();
            List<double> resultVals = new List<double>();

            topology.Add(3);
            topology.Add(2);
            topology.Add(1);

            NeuralNet myNet = new NeuralNet(topology);

            myNet.feedForward(inputVals);
            myNet.backProp(targetVals);
            myNet.getResults(resultVals);
            */
        }

        static void CreateTrainingFile(string filename)
        {
            Random rnd = new Random();
            StreamWriter file = new StreamWriter(new FileStream(filename + ".txt", FileMode.Create));
            string fileData = "topology: 2 4 1 \n";
            for (int i = 2000; i > 0; --i)
            {
                int n1 = (int)(2.0 * rnd.NextDouble());
                int n2 = (int)(2.0 * rnd.NextDouble());
                int t = n1 ^ n2;
                fileData += "in: " + n1 + ".0 " + n2 + ".0\n";
                fileData += "out: " + t + ".0\n";
            }
            file.Write(fileData);
            file.Close();
        }

        static void showVectorVals(string label, List<double> v)
        {
            Console.Write(label + " ");
            for (int i = 0; i < v.Count; ++i) {
                Console.Write(Math.Round((1/v[i]),2) + " ");
            }

            Console.Write("\n");
        }
    }
}
