package cz.pv021.neuralnets.functions;

import static java.lang.Math.*;

/**
 *
 * @author  Lukáš Daubner
 * @since   2016-12-10
 * @version 2016-12-13
 */
public class Softsign implements ActivationFunction {
    @Override
    public double apply (double innerPotential) {
        return innerPotential / (1 + abs(innerPotential));
    }

    @Override
    public double derivative (double innerPotential) {
        // (1+|x| - x*sgn(x)) / (1+|x|)^2
        return (1 + abs(innerPotential) - innerPotential * signum(innerPotential)) / pow(1 + abs(innerPotential), 2);
    }
}
