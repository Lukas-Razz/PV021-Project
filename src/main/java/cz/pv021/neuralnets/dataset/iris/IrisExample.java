package cz.pv021.neuralnets.dataset.iris;

import cz.pv021.neuralnets.dataset.ExampleImpl;
import cz.pv021.neuralnets.dataset.SimpleExample;

/**
 * @author  Josef Plch
 * @since   2016-12-08
 * @version 2016-12-15
 */
public class IrisExample extends ExampleImpl <double[], IrisClass> implements SimpleExample <IrisClass> {
    public IrisExample (double[] attributes, IrisClass irisClass) {
        super (attributes, irisClass);
    }
}
