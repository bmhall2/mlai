using System.Numerics;

namespace GradientDescent;

public static class Formulas
{
    public static double Predict(double[] x, double[] w, double b)
    {
        return Dot(x, w) + b;
    }

    public static double Dot(double[] x, double[] y)
    {
        return Vector.Dot(new Vector<double>(x), new Vector<double>(y));
    }

    public static double Cost(double[][] x, double[] y, double[] w, double b)
    {
        double cost = 0;
        double m = x.Length;

        for (int i = 0; i < m; i++)
        {
            cost += Math.Pow(Dot(x[i], w) + b - y[i], 2);
        }

        return cost / (2f * m);
    } 

    public static (double[], double) Gradient(double[][] x, double[] y, double[] w, double b)
    {
        var m = x.Length;
        var n = x[0].Length;

        var dj_dw = new double[n];
        double dj_db = 0;


        for (int i = 0; i < m; i++)
        {
            var err = Dot(x[i], w) + b - y[i];
            for (int j = 0; j < n; j++)
            {
                dj_dw[j] += err * x[i][j];
            }
            dj_db += err;
        }

        for (int j = 0; j < n; j++)
        {
            dj_dw[j] = dj_dw[j] / m;
        }

        dj_db = dj_db / m;

        return (dj_dw, dj_db);
    }

    public static (double[] w, double b) GradientDesc(double[][] x, double[] y, double[] w_in, double b_in, double alpha, int iterations)
    {
        var w = w_in;
        var b = b_in;

        for (int i = 0; i < iterations; i++)
        {
            var gradient = Gradient(x, y, w, b);
            for (int j = 0; j < w.Length; j++)
            {
                w[j] = w[j] - alpha * gradient.Item1[j];
            }
            b = b - alpha * gradient.Item2;

            if (i % Math.Ceiling(iterations / (double)10) == 0)
            {
                Console.WriteLine($"Iteration: {i}: Cost={Cost(x, y, w, b)}");
            }
        }

        return (w, b);
    }
}
