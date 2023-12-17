using GradientDescent;

internal class Program
{
    private static void Main(string[] args)
    {
        var x_train = new double[][] { new double[] { 2104, 5, 1, 45 }, new double[] { 1416, 3, 2, 40 }, new double[] { 852, 2, 1, 35 } };
        var y_train = new double[] { 460, 232, 178 };

        // var b_init = 785.1811367994083;
        // var w_init = new double[] { 0.39133535, 18.75376741, -53.36032453, -26.42131618 };

        // Console.WriteLine($"Prediction: {Formulas.Predict(x_train[0], w_init, b_init)}");
        // Console.WriteLine($"Cost init: {Formulas.Cost(x_train, y_train, w_init, b_init)}");

        // var gradient = Formulas.Gradient(x_train, y_train, w_init, b_init);
        // var dj_dw = gradient.Item1;
        // var dj_db = gradient.Item2;

        // Console.WriteLine($"Gradient dj_db init: {dj_db}");
        // Console.Write("Gradient dj_dw: [ ");
        // foreach (var d in dj_dw)
        // {
        //     Console.Write($"{d} ");
        // }
        // Console.WriteLine("]");

        var b_init = 0;
        var w_init = new double[] { 0, 0, 0, 0 };

        var gradient_desc = Formulas.GradientDesc(x_train, y_train, w_init, b_init, 0.0000005, 1000);
        Console.WriteLine($"Gradient Desc b : {gradient_desc.b}");
        Console.Write("Gradient Desc w: [ ");
        foreach (var w in gradient_desc.w)
        {
            Console.Write($"{w} ");
        }
        Console.WriteLine("]");

        Console.WriteLine($"Prediction: {Formulas.Predict(x_train[0], gradient_desc.w, gradient_desc.b)} || Target: {y_train[0]}");
        Console.WriteLine($"Prediction: {Formulas.Predict(x_train[1], gradient_desc.w, gradient_desc.b)} || Target: {y_train[1]}");
        Console.WriteLine($"Prediction: {Formulas.Predict(x_train[2], gradient_desc.w, gradient_desc.b)} || Target: {y_train[2]}");
    }
}