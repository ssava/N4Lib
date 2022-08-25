namespace N4Lib;

public sealed class WeightsInitializer
{
    private Random _rng;
    public WeightsInitializer()
        => _rng = new Random();

    // randn(layer_size, layer_size-1)
    public double Next()
        => _rng.NextDouble();

    public double Next(IActivationFunction layerActivationFunction, int layerSize)
        => Next() * layerActivationFunction.OptimizationFactor(layerSize);
}