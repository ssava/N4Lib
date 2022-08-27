namespace N4Lib;

public sealed class Network
{
    public sealed record Configuration
    {
        public int InputSize { get; } = 1;

        public int HiddenLayerSize { get; } = 1;

        public int OutputSize { get; } = 1;

        public IActivationFunction ActivationFunction { get; }
        public WeightsInitializer WeightsInitializer { get; }

        public Configuration(int input_size,
                             int hidden_size,
                             int output_size,
                             IActivationFunction activation,
                             WeightsInitializer weights)
        {
            InputSize = input_size;
            HiddenLayerSize = hidden_size;
            OutputSize = output_size;
            ActivationFunction = activation;
            WeightsInitializer = weights;
        }
    }

    private Graph<Neuron> _network;

    private Configuration _topology;

    private IEnumerable<Layer> _layers;

    public Network(Configuration topology)
    {
        _topology = topology;

        _network = new();

        _layers = new List<Layer>();

        var inputLayer = Layer.Input(_topology.InputSize);

        _layers.Append(inputLayer);

        var hiddenLayer = Layer.Create(_topology.HiddenLayerSize,
                                       _topology.ActivationFunction);

        _layers.Append(hiddenLayer);

        var outputLayer = Layer.Output(_topology.OutputSize);

        _layers.Append(outputLayer);

        // FIXME: Generalize connections
        Connect(inputLayer, hiddenLayer, _topology.WeightsInitializer);
        Connect(hiddenLayer, outputLayer, _topology.WeightsInitializer);
    }

    private void Connect(Layer src, Layer dst, WeightsInitializer weightsInitializer)
    {
        var source_nodes = src.Neurons.Select(x => new Graph<Neuron>.Node(x));
        var destination_nodes = dst.Neurons.Select(x => new Graph<Neuron>.Node(x));

        foreach (var from_neuron in source_nodes)
        {
            foreach (var to_neuron in destination_nodes)
            {
                var randomWeight = weightsInitializer.Next(src.Function, src.Size);
                
                _network.AddEdge(from_neuron, to_neuron, randomWeight);
            }
        }
    }

    public async Task<IEnumerable<double>> FeedAsync(params double[] values)
    {
        if (values.Count() != _topology.InputSize)
            throw new ArgumentException("The length of data does not match network input");

        // Set input layer neurons value.
        SetInputLayerNeurons(values);

        // Start propagation until output layer neurons
        await PropagateAsync();

        // Read values from last neurons layer (output)
        return _layers.Last().Neurons.Select(n => n.Value);
    }

    private void SetInputLayerNeurons(IEnumerable<double> values)
    {
        foreach (int i in Enumerable.Range(0, values.Count() - 1))
        {
            _layers.First().Neurons.ElementAt(i).Value = values.ElementAt(i);
        }
    }

    private Task PropagateAsync()
    {
        foreach (var layer in _layers)
        {
            var neuronConnectionByDestination = _network.Edges.Where(e => layer.Neurons.Contains(e.From.Item))
                                                              .GroupBy(e => e.To);

            foreach(var neuronEdges in neuronConnectionByDestination)
                neuronEdges.Key.Item.Value = neuronEdges.Sum(e => e.From.Item.Activate() * e.Weight);
        }

        return Task.CompletedTask;
    }
}