package cz.pv021.neuralnets.dataset;

import cz.pv021.neuralnets.dataset.iris.IrisClass;

/**
 * @author  Lukáš Daubner
 * @since   2016-12-10
 * @version 2016-12-10
 */
public interface Example {

    double[] getAttributes();

    IrisClass getIrisClass(); //TODO: Odstranit závyslost na Iris
    
}
