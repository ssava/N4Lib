namespace N4Lib;

public sealed class NetworkTrainer
{
    public ICollection<ICollection<double>> Dataset { get; }

    public LossFunction LossFunction { get; }

    public Network Network { get; }

    public NetworkTrainer(ICollection<ICollection<double>> dataset,
                          LossFunction lossFunction,
                          Network network)
        => (this.Dataset, this.LossFunction, this.Network) = (dataset, lossFunction, network);

    public double Train(int numberOfEpochs, double learningRate)
    {
        return 0.0f;
    }

    public double StepEpoch(double learningRate)
    {
        return 0.0f;
    }
}