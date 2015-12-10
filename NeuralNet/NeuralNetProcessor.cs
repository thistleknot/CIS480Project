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
            //for tracking data (csv file)
            Dictionary<int, double> trainListTargets = new Dictionary<int, double>();
            Dictionary<int, double> testListTargets = new Dictionary<int, double>();
            Dictionary<int, double> testListOutputs = new Dictionary<int, double>();
            Dictionary<int, double> testOutputsDiffTrainTarget = new Dictionary<int, double>();
            Dictionary<int, double> testTargetDiffTrainTarget = new Dictionary<int, double>();

            int fileCount = (from file in Directory.EnumerateFiles(@".\", "train*.txt")
                             select file).Count();
            
            for (int x = 0; x < fileCount; x++)
            {

                //CreateTrainingFile("test");

                //need to include iterations
                string train = "train" + x.ToString("D3") + ".txt";
                string test = "test" + x.ToString("D3") + ".txt";
                double lastTrainError = 1;

                Console.WriteLine("file {0} out of {1}", x+1, fileCount);

                TrainingData trainData = new TrainingData(train);
                TrainingData testData = new TrainingData(test);

                // e.g., { 3, 2, 1 }
                //training
                List<int> topology = new List<int>();                
                List<double> inputVals = new List<double>();
                List<double> targetVals = new List<double>();
                List<double> resultVals = new List<double>();

                trainData.getTopology(topology);

                NeuralNet myNet = new NeuralNet(topology);

                //testing
                List<int> topologyTest = new List<int>();
                List<double> testInputVals = new List<double>();
                List<double> testTargetVals = new List<double>();
                List<double> testResultVals = new List<double>();

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

                    Debug.Assert(targetVals.Count == topology.Last());
                    myNet.backProp(targetVals);
                    // Report how well the training is working, average over recent
                    // samples:
                    //Console.WriteLine("Net recent average error: " + myNet.RecentAverageError);
                    lastTrainError = myNet.RecentAverageError;
                    //if ((trainingPass > 100) && (myNet.RecentAverageError < .03)) break; 
                }

                trainData.close();

                if (lastTrainError > .9)
                {
                    Console.WriteLine("fail");
                    x = x - 1;
                }
                else
                {
                    //desired output
                    showVectorVals("Last Training Output:", resultVals);
                    //trainData.getTargetOutputs(targetVals);
                    //showVectorVals("Last Training Targets:", finalTrainTargets);
                    Console.WriteLine("Net recent average error: " + (lastTrainError.ToString()));

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
                        trainListTargets.Add(x, testInputVals[testInputVals.Count()-1]);
                        testListTargets.Add(x, targetVals[targetVals.Count() - 1]);
                        testListOutputs.Add(x, resultVals[resultVals.Count()-1]);
                    }


                }
                testData.close();


            }

            Double directionSuccessRate = 0;

            using (System.IO.StreamWriter file = new System.IO.StreamWriter("outputFile.csv", true))
            {
                Console.WriteLine("trainTarget,testTarget,testOutput");
                file.WriteLine("trainTarget,testTarget,testOutput");

                for (int x = 0; x < testListTargets.Count(); x++)
                {
                    file.WriteLine("{0},{1},{2}", Math.Round(1 / trainListTargets[x], 2), Math.Round(1 / testListTargets[x], 2), Math.Round(1 / testListOutputs[x], 2));
                    Console.WriteLine("{0},{1},{2}", Math.Round(1 / trainListTargets[x], 2), Math.Round(1 / testListTargets[x], 2), Math.Round(1 / testListOutputs[x], 2));

                    testOutputsDiffTrainTarget.Add(x, (testListOutputs[x] - trainListTargets[x]));
                    testTargetDiffTrainTarget.Add(x, (testListTargets[x] - trainListTargets[x]));

                    if ((testOutputsDiffTrainTarget[x]* testTargetDiffTrainTarget [x])> 0)
                    {
                        directionSuccessRate++;
                    }


                }
                directionSuccessRate = directionSuccessRate / testListTargets.Count();
                Console.WriteLine("successRate: {0}", directionSuccessRate);
                file.WriteLine("successRate: {0}", directionSuccessRate);
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
