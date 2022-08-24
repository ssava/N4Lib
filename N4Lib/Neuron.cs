namespace N4Lib;

public sealed record Neuron
{
    private readonly IActivationFunction _func;

    private Neuron(IActivationFunction func, bool hasBias)
		  => (this._func, HasBias) = (func, hasBias);

    public static Neuron Build(IActivationFunction actFunc, bool hasBias = false)
		  => new(actFunc, hasBias);

    /// <summary>
    /// Neuron value.
    /// </summary>
    /// <remarks>This value should be set as SUM of weighted inputs</remarks>
    public double Value { get; set; } = 0.0f;

    public bool HasBias { get; init; }

    public double Activate()
		  => this._func.Activate(Value);

    public double Prime()
        => this._func.Prime(Value);
}
