package cz.pv021.neralnets.initialization;

import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.Random;

/**
 * Initialization from uniform distribution
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-10
 * @version 2016-12-10
 */
public class UniformInitialization implements Initialization {
    
    @Override
    public LayerParameters initializeTANHLike(LayerParameters parameters, int seed) {
        Random random = new Random (seed);
        for (int i = 0; i < parameters.getWeights().length; i++) {
            for (int j = 0; j < parameters.getWeights()[i].length; j++) {
                parameters.getWeights()[i][j] = random.nextDouble() * (random.nextBoolean() ? 1 : -1);
            }
        }
        for (int i = 0; i < parameters.getBias().length; i++) {
            parameters.getBias()[i] = 0;
        }
        return parameters;
    }
    
    @Override
    public LayerParameters initializeSIGMOIDLike(LayerParameters parameters, int seed) {
        Random random = new Random (seed);
        for (int i = 0; i < parameters.getWeights().length; i++) {
            for (int j = 0; j < parameters.getWeights()[i].length; j++) {
                parameters.getWeights()[i][j] = random.nextDouble();
            }
        }
        for (int i = 0; i < parameters.getBias().length; i++) {
            parameters.getBias()[i] = 0;
        }
        return parameters;
    }
}
