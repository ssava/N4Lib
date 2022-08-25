namespace N4Lib;

public sealed class WeightsInitializer
{
    private Random _rng;
    public WeightsInitializer()
        => _rng = new Random();

    // randn(layer_size, layer_size-1)
    public double Next(IActivationFunction layerActivationFunction, int layerSize)
        => _rng.NextDouble() * layerActivationFunction.OptimizationFactor(layerSize);
}