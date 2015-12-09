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

            int fileCount = (from file in Directory.EnumerateFiles(@".\", "train*.txt", SearchOption.AllDirectories)
                             select file).Count();

            

            for (int x = 0; x < fileCount; x++)
            {

                //CreateTrainingFile("test");

                //need to include iterations
                string train = "train" + x.ToString("D3") + ".txt";
                string test = "test" + x.ToString("D3") + ".txt";
                string lastTrainError = "";

                Console.WriteLine("file {0} out of {1}", x+1, fileCount);

                TrainingData trainData = new TrainingData(train);
                TrainingData testData = new TrainingData(test);

                // e.g., { 3, 2, 1 }
                List<int> topology = new List<int>();
                List<int> topologyTest = new List<int>();
                List<double> inputVals = new List<double>();
                List<double> testInputVals = new List<double>();
                List<double> targetVals = new List<double>();
                List<double> resultVals = new List<double>();
                trainData.getTopology(topology);

                NeuralNet myNet = new NeuralNet(topology);

                List<double> finalTrainOutput = new List<double>();
                List<double> finalTrainTargets = new List<double>();

                List<double> testInputs = new List<double>();
                List<double> testTargets = new List<double>();
                List<double> testOutputs = new List<double>();

                int trainingPass = 0;
                Console.WriteLine("train pass");
                while (!trainData.isEof())
                {
                    ++trainingPass;
                    //Console.Write("\nPass " + trainingPass);
                    // Get new input data and feed it forward:
                    if (trainData.getNextInputs(inputVals) != topology[0])
                    {
                        break;
                    }
                    //showVectorVals("Inputs:", inputVals);
                    myNet.feedForward(inputVals);
                    // Collect the net's actual output results:
                    myNet.getResults(resultVals);
                    //showVectorVals("Outputs:", resultVals);
                    // Train the net what the outputs should have been:
                    trainData.getTargetOutputs(targetVals);
                    //showVectorVals("Targets:", targetVals);

                    //new
                    finalTrainTargets = targetVals;

                    Debug.Assert(targetVals.Count == topology.Last());
                    myNet.backProp(targetVals);
                    // Report how well the training is working, average over recent
                    // samples:
                    //Console.WriteLine("Net recent average error: " + myNet.RecentAverageError);
                    lastTrainError = "Net recent average error: " + myNet.RecentAverageError;
                    //if ((trainingPass > 100) && (myNet.RecentAverageError < .03)) break; 
                }

                //desired output
                showVectorVals("Last Training Output:", resultVals);
                //trainData.getTargetOutputs(targetVals);
                showVectorVals("Last Training Targets:", finalTrainTargets);
                Console.WriteLine("Net recent average error: " + lastTrainError);

                trainData.close();

                testData.getTopology(topologyTest);

                Console.WriteLine("test pass");

                while (!testData.isEof())
                {
                    
                    if (testData.getNextInputs(testInputVals) != topology[0])
                    {
                        break;
                    }
                    
                    //test data
                    //showVectorVals("Inputs:", testInputVals);
                    myNet.feedForward(testInputVals);
                    // Collect the net's actual output results:
                    myNet.getResults(resultVals);
                    showVectorVals("Test Outputs:", resultVals);
                    // Train the net what the outputs should have been:
                    testData.getTargetOutputs(targetVals);
                    showVectorVals("Test Targets:", targetVals);
                    Debug.Assert(targetVals.Count == topology.Last());
                }
                testData.close();

                //Console.WriteLine("\nDone");



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
