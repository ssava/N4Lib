namespace N4Lib;

public sealed record ActivationFunction : IActivationFunction
{
    public static readonly ActivationFunction Identity
		= new(x => x, x => 1);

    public static readonly ActivationFunction LeakyReLU
		= new(x => x * (x < 0 ? 0.01 : 1), x => x < 0 ? 0.01 : 1);

    public static readonly ActivationFunction None
              = new(x => x, x => x);

    public static readonly ActivationFunction ReLU
              = new(x => x < 0 ? 0 : x, x => x < 0 ? 0 : 1);

    public static readonly ActivationFunction Sigmoid
		= new(
				func: x => 1 / (1 + Math.Exp(-x)),
				prime: x => (1 / (1 + Math.Exp(-x)) * (1 - (1 / (1 + Math.Exp(-x)))))
		);

    public static readonly ActivationFunction SoftPlus
		= new(x => Math.Log(1 + Math.Exp(x)), x => 1 / (1 + Math.Exp(-x)));

    public static readonly ActivationFunction SoftSign
		= new(
			func: x => x / (1 + Math.Abs(x)),
			prime: x => 1 / Math.Pow(1 + Math.Abs(x), 2)
		);

    private static readonly ActivationFunction Tanh
		= new(x => Math.Tanh(x), x => Math.Pow(Math.Tanh(x), 2));

    private Func<double, double> _activate { get; }

    private Func<double, double> _prime { get; }

    private ActivationFunction
        (Func<double, double> func, Func<double, double> prime)
            => (_activate, _prime) = (func, prime);

	public double Activate(double x)
		=> this._activate(x);

	public double Prime(double x)
		=> this._prime(x);
}
