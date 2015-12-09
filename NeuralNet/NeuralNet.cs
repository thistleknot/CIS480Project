using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using Layer = System.Collections.Generic.List<NeuralNet.Neuron>;

namespace NeuralNet
{
    class NeuralNet
    {
        List<Layer> m_layers; 
        private double m_error;
        private double m_recentAverageError;
        private double m_recentAverageSmoothingFactor = 100.0;
        public double RecentAverageError { get { return m_recentAverageError; } }

        public NeuralNet(List<int> topology)
        {
            m_layers = new List<Layer>();
            int numLayers = topology.Count;
            for (int layerNum = 0; layerNum < numLayers; layerNum++)
            {
                m_layers.Add(new Layer());
                int numOutputs = layerNum == topology.Count - 1 ? 0 : topology[layerNum + 1];
                for (int neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
                {
                    m_layers[layerNum].Add(new Neuron(numOutputs, neuronNum));
                }
                Console.WriteLine("Constructed a Layer");
                // force the bias node's output value to 1.0. It's the last neuron created above
                m_layers.Last().Last().OutputVal = 1.0;
            }
        }

        public void feedForward(List<double> inputVals)
        {
            Debug.Assert(inputVals.Count == m_layers[0].Count - 1);
            Console.WriteLine("Running NeuralNet.feedForward");
            // Assign (latch) the input values into the input neurons
            for (int input = 0; input < inputVals.Count; ++input)
            {
                m_layers[0][input].OutputVal = inputVals[input];
            }

            // forward propagation
            for (int layerNum = 1; layerNum < m_layers.Count; ++layerNum)
            {
                Layer prevLayer = m_layers[layerNum - 1];
                for (int node = 0; node < m_layers[layerNum].Count - 1; ++node)
                {
                    m_layers[layerNum][node].feedForward(prevLayer);
                }
            }

        }

        public void backProp(List<double> targetVals)
        {
            // Calculate overall net error (RMS of output neuron errors)
            Layer outputLayer = m_layers.Last();
            m_error = 0.0;

            for (int node = 0; node < outputLayer.Count - 1; ++node)
            {
                double delta = targetVals[node] - outputLayer[node].OutputVal;
                m_error += delta * delta;
            }
            m_error /= outputLayer.Count - 1; // get average error squared
            m_error = Math.Sqrt(m_error); // RMS

            // Implement a recent average measurement:

            m_recentAverageError =
                (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
                / (m_recentAverageSmoothingFactor + 1.0);

            // Calculate output layer gradients

            for (int node = 0; node < outputLayer.Count - 1; ++node)
            {
                outputLayer[node].calcOutputGradients(targetVals[node]);
            }

            // Calculate gradients on hidden layers

            for (int layerNum = m_layers.Count - 2; layerNum > 0; --layerNum)
            {
                Layer hiddenLayer = m_layers[layerNum];
                Layer nextLayer = m_layers[layerNum + 1];
                for (int node = 0; node < hiddenLayer.Count; ++node)
                {
                    hiddenLayer[node].calcHiddenGradients(nextLayer);
                }
            }

            // For all layers from outputs to first hidden layer,
            // update connection weights

            for (int layerNum = m_layers.Count - 1; layerNum > 0; --layerNum)
            {
                Layer layer = m_layers[layerNum];
                Layer prevLayer = m_layers[layerNum - 1];

                for (int node = 0; node < layer.Count - 1; ++node)
                {
                    layer[node].updateInputWeights(prevLayer);
                }
            }
        }

        public void getResults(List<double> resultVals)
        {
            Console.WriteLine("Running NeuralNet.getResults");
            resultVals.Clear();

            for (int node = 0; node < m_layers.Last().Count - 1; ++node)
            {
                resultVals.Add(m_layers.Last()[node].OutputVal);
            }
        }


    }
}
