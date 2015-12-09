using System;
using System.Collections.Generic;
using Layer = System.Collections.Generic.List<NeuralNet.Neuron>;

namespace NeuralNet
{
    struct Connection
    {
        public double Weight;
        public double DeltaWeight;

        public Connection(double weight = 0.0, double deltaWeight = 0.0)
        {
            Weight = weight;
            DeltaWeight = deltaWeight;
        }
    }

    class Neuron
    {


        private static double eta = 0.15;   // [0.0 ... 1.0] overall     net learning rate;
        private static double alpha = 0.5;  // [0.0 ... n] multiplier of last weight change (momentum)
        //private static Random rnd = new Random();

        Random rnd = new Random();

        private double m_outputVal = 0.0;
        List<Connection> m_outputWeights;
        private int m_myIndex;
        private double m_gradient;

        //private static double RandomWeight()
        private double RandomWeight()
        {
            return rnd.NextDouble();
        }

        public Neuron(int numOutputs, int myIndex)
        {
            m_outputWeights = new List<Connection>();
            for (int count = 0; count < numOutputs; ++count)
            {
                m_outputWeights.Add(new Connection(RandomWeight()));
            }
            m_myIndex = myIndex;
            //Console.WriteLine("Constructed a Neuron");
        }

        public double OutputVal
        {
            get { return m_outputVal; }
            set { m_outputVal = value; }
        }

        public void feedForward(Layer prevLayer)
        {
            double sum = 0.0;

            // Sum the previous layer's outputs (which are our inputs)
            // Include the bias node from the previous layer.

            for (int node = 0; node < prevLayer.Count; ++node)
            {
                sum += prevLayer[node].m_outputVal * 
                    prevLayer[node].m_outputWeights[m_myIndex].Weight;
            }
            m_outputVal = Neuron.TransferFunction(sum);
        }

        private static double TransferFunction(double x)
        {
            // tanh - output range [-1.0 ... 1.0]
            return Math.Tanh(x);
        }

        private static double TransferFunctionDerivative(double x)
        {
            // tanh derivative
            return 1.0 - x * x;
        }

        public void updateInputWeights(Layer prevLayer)
        {
            // The weights to be updated are in the Connection container
            // in the neurons in the preceding layer

            for (int node = 0; node < prevLayer.Count; ++node)
            {
                Neuron neuron = prevLayer[node];
                double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].DeltaWeight;

                double newDeltaWeight =
                    // Individual input, magnified by the gradient and train rate:
                    eta
                    * neuron.OutputVal
                    * m_gradient
                    // Also add momentum = a fraction of the previous delta weight
                    + alpha
                    * oldDeltaWeight;
                Connection connection = neuron.m_outputWeights[m_myIndex];
                connection.DeltaWeight = newDeltaWeight;
                connection.Weight += newDeltaWeight;
                neuron.m_outputWeights[m_myIndex] = connection;
            }


        }

        public void calcHiddenGradients(Layer nextLayer)
        {
            double dow = sumDOW(nextLayer);
            m_gradient = dow * TransferFunctionDerivative(m_outputVal);
        }

        private double sumDOW(Layer nextLayer)
        {
            double sum = 0.0;

            // Sum our contributions of the errors at the nodes we feed

            for (int node = 0; node < nextLayer.Count - 1; ++node)
            {
                sum += m_outputWeights[node].Weight * nextLayer[node].m_gradient;
            }

            return sum;
        }

        public void calcOutputGradients(double targetVal)
        {
            double delta = targetVal - m_outputVal;
            m_gradient = delta * TransferFunctionDerivative(m_outputVal);
        }
    }
}
