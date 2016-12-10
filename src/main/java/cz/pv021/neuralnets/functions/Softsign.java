package cz.pv021.neuralnets.functions;

import static java.lang.Math.*;



/**
 *
 * @author  Lukáš Daubner
 * @since   2016-12-10
 * @version 2016-12-10
 */
public class Softsign implements ActivationFunction {
    @Override
    public double apply (double innerPotencial) {
        return innerPotencial / (1 + abs(innerPotencial));
    }

    @Override
    public double derivative (double innerPotencial) {
        // (1+|x| - x*sgn(x)) / (1+|x|)^2
        return (1+abs(innerPotencial)-innerPotencial*signum(innerPotencial)) / pow(1+abs(innerPotencial), 2);
    }
}
