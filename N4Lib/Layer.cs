namespace N4Lib;

public sealed record Layer
{
    public static Layer Input(int size)
        => new Layer(size, ActivationFunction.None);

    public static Layer Create(int size, IActivationFunction activationFunction)
        => new Layer(size, activationFunction);

    public static Layer Output(int size)
        => new Layer(size, ActivationFunction.None);

    public IEnumerable<Neuron> Neurons { get; }

    public IActivationFunction Function { get; }

    public int Size { get; }

    private Layer(int size, IActivationFunction activationFunction)
    {
        Size = size;
        Function = activationFunction;
        Neurons = Enumerable.Range(1, size)
                            .Select(x => Neuron.Build(activationFunction));
    }
}