namespace N4Lib;

public sealed class NetworkTrainer
{
    public ICollection<ICollection<double>> Dataset { get; }

    public LossFunction LossFunction { get; }

    public Network Network { get; }

    public NetworkTrainer(ICollection<ICollection<double>> dataset,
                          LossFunction lossFunction,
                          Network network)
    {
        if (!dataset.Any())
            throw new ArgumentException("Empty dataset!");

        if (dataset.First().Count() != network.InputSize + network.OutputSize)
            throw new ArgumentException("Invalid dataset size!");

        this.Dataset = dataset;
        this.LossFunction = lossFunction;
        this.Network = network;
    }

    public double Train(int numberOfEpochs, double learningRate)
    {
        return 0.0f;
    }

    public async Task<double> StepEpochAsync(double learningRate)
    {
        foreach (var example in Dataset)
        {
            var input = example.Take(Network.InputSize);
            var expectedOutput = example.Skip(Network.InputSize)
                                        .Take(Network.OutputSize);

            var output = await Network.FeedAsync(input.ToArray() ?? Array.Empty<double>());

            // Calculate output delta
            var delta = LossFunction.Loss(expectedOutput, output);

            // Backpropagate using output delta
            await BackpropagateAsync(delta);
        }
        
        return 0.0f;
    }

    private Task BackpropagateAsync(double loss)
    {
        return Task.CompletedTask;
    }
}