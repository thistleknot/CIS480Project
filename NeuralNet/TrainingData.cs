using System;
using System.IO;
using System.Collections.Generic;

namespace NeuralNet
{
    class TrainingData
    {
        public bool isEof() { return m_trainingDataFile.EndOfStream; }
        private StreamReader m_trainingDataFile;

        public void getTopology(List<int> topology)
        {
            string line;

            StreamReader fileReader = m_trainingDataFile;
            line = fileReader.ReadLine();

            string[] ss = line.Split(' ','\n');
            if (isEof() || ss[0].CompareTo("topology:") != 0)
            {
                throw new Exception("I'm broke!");
            }
            foreach (string label in ss)
            {
                if (label.CompareTo("") != 0 && label.CompareTo("topology:") != 0)
                {
                    int num;
                    int.TryParse(label, out num);
                    topology.Add(num);
                }
             }
        }

        public TrainingData(string filename)
        {
            m_trainingDataFile = new StreamReader(new FileStream(filename, FileMode.Open));
        }

        public long getNextInputs(List<double> inputVals)
        {
            inputVals.Clear();

            string line;
            StreamReader fileReader = m_trainingDataFile;
            line = fileReader.ReadLine();
            string[] ss = line.Split(' ');

            foreach (string label in ss)
            {
                if (label.CompareTo("in:") != 0)
                {
                    double oneValue;
                    double.TryParse(label, out oneValue);
                    inputVals.Add(oneValue);
                }
            }

            return inputVals.Count;
        }

        public long getTargetOutputs(List<double> targetOutputVals)
        {
            targetOutputVals.Clear();

            string line;
            StreamReader fileReader = m_trainingDataFile;
            line = fileReader.ReadLine();
            string[] ss = line.Split(' ');

            foreach (string label in ss)
            {
                if (label.CompareTo("out:") != 0)
                {
                    double oneValue;
                    double.TryParse(label, out oneValue);
                    targetOutputVals.Add(oneValue);
                }
            }
            return targetOutputVals.Count;
        }

        public void close()
        {
            m_trainingDataFile.Close();
        }
    }
}
