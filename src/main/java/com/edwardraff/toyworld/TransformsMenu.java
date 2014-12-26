/*
 * Copyright (C) 2014 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.edwardraff.toyworld;

import java.awt.Component;
import java.awt.HeadlessException;
import javax.swing.Action;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JSeparator;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.datatransform.DataTransformFactory;
import jsat.datatransform.DataTransformProcess;
import jsat.datatransform.LinearTransform;
import jsat.datatransform.PCA;
import jsat.datatransform.PolynomialTransform;
import jsat.datatransform.WhitenedZCA;
import jsat.datatransform.kernel.Nystrom;
import jsat.distributions.kernels.RBFKernel;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.text.GreekLetters;

/**
 * Convenience class for adding all the transforms we wish to use
 * 
 * @author edwardraff
 */
public class TransformsMenu extends JMenu
{
    private final Component parent;
    /**
     * The main data transform process
     */
    private DataTransformProcess dataTransformProcess;
    /**
     * Menu used to provide a list of the currently queued transforms 
     */
    private JMenu currentTransforms;
    private static final JLabel noTransformsLabel = new JLabel("No Transforms Currently");

    public TransformsMenu(Component parent, String s, boolean b)
    {
        super(s, b);
        this.parent = parent;
        initiate();
    }

    public TransformsMenu(Component parent, String s)
    {
        super(s);
        this.parent = parent;
        initiate();
    }

    public TransformsMenu(Component parent, Action a)
    {
        super(a);
        this.parent = parent;
        initiate();
    }

    public TransformsMenu(Component parent)
    {
        super();
        this.parent = parent;
        initiate();
    }
    
    private void initiate()
    {
        dataTransformProcess = new DataTransformProcess();
        currentTransforms = new JMenu("Current Transforms");
        currentTransforms.add(noTransformsLabel);
        
        
        add(newMenuItem("Add Bias", (e) ->
        {
            dataTransformProcess.addTransform(new DataTransformFactory()
            {

                @Override
                public DataTransform getTransform(DataSet dataset)
                {
                    return new DataTransform()
                    {

                        @Override
                        public DataPoint transform(DataPoint dp)
                        {
                            Vec orig = dp.getNumericalValues();
                            Vec newVec = new DenseVector(orig.length() + 1);
                            for (IndexValue iv : orig)
                                newVec.set(iv.getIndex(), iv.getValue());
                            newVec.set(orig.length(), 1.0);

                            return new DataPoint(newVec, dp.getCategoricalValues(), dp.getCategoricalData());
                        }

                        @Override
                        public DataTransform clone()
                        {
                            return this;
                        }
                    };
                }

                @Override
                public DataTransformFactory clone()
                {
                    return this;
                }
            });
            
            addTransformName("Add Bias");
        }));
        add(newMenuItem("Linear Rescaling", (e)->
        {
            String s = JOptionPane.showInputDialog(parent, "Please Specify the range to trasform the input to as \"low, high\"", "Linear Rescaling", JOptionPane.QUESTION_MESSAGE);

            try
            {
                String[] vals = s.replace(",", " ").split("\\s+");
                double min = Double.parseDouble(vals[0]);
                double max = Double.parseDouble(vals[1]);
                dataTransformProcess.addTransform(new LinearTransform.LinearTransformFactory(min, max));
                addTransformName("Linear Rescaling [" + min + ", " + max + "]");
            }
            catch (Exception ex)
            {
                JOptionPane.showMessageDialog(parent, "Error adding the transform: " + ex.getMessage(), "Linear Rescaling: Error", JOptionPane.ERROR_MESSAGE);
            }
        }));
        add(newMenuItem("PCA", (e)->
        {
            dataTransformProcess.addTransform(new PCA.PCAFactory(2));
            addTransformName("PCA");
        }));
        add(newMenuItem("Whitening", (e)->
        {
            dataTransformProcess.addTransform(new WhitenedZCA.WhitenedZCATransformFactory(1e-4));
            addTransformName("Whitening");
        }));
        add(newMenuItem("Nystrom RBF", (e)->
        {
            String s = JOptionPane.showInputDialog(parent, "Please specify the number of dimensions for the transform", "Nystrom RBF", JOptionPane.QUESTION_MESSAGE);

            try
            {
                int empiricalDim = Integer.parseInt(s);
                
                s = JOptionPane.showInputDialog(parent, "Please specify the RBF width for the transform", "Nystrom RBF", JOptionPane.QUESTION_MESSAGE);
                double width = Double.parseDouble(s);
                
                dataTransformProcess.addTransform(new Nystrom.NystromTransformFactory(new RBFKernel(width), empiricalDim, Nystrom.SamplingMethod.KMEANS, 0.0001, false));
                addTransformName("Nystrong RBF (Dim=" + empiricalDim + ", " + GreekLetters.sigma + "=" + width+")");
            }
            catch (Exception ex)
            {
                JOptionPane.showMessageDialog(parent, "Error adding the transform: " + ex.getMessage(), "Nystrom RBF: Error", JOptionPane.ERROR_MESSAGE);
            }
        }));
        add(newMenuItem("Random Kitchen Sinks RBF", (e)->
        {
            String s = JOptionPane.showInputDialog(parent, "Please specify the number of dimensions for the transform", "Random Kitchen Sinks RBF", JOptionPane.QUESTION_MESSAGE);

            try
            {
                int empiricalDim = Integer.parseInt(s);
                
                s = JOptionPane.showInputDialog(parent, "Please specify the RBF width for the transform", "Random Kitchen Sinks RBF", JOptionPane.QUESTION_MESSAGE);
                double width = Double.parseDouble(s);
                
                dataTransformProcess.addTransform(new Nystrom.NystromTransformFactory(new RBFKernel(width), empiricalDim, Nystrom.SamplingMethod.KMEANS, 0.0001, false));
                addTransformName("Random Kitchen Sinks RBF (Dim=" + empiricalDim + ", " + GreekLetters.sigma + "=" + width+")");
            }
            catch (NumberFormatException | HeadlessException ex)
            {
                JOptionPane.showMessageDialog(parent, "Error adding the transform: " + ex.getMessage(), "Random Kitchen Sinks RBF: Error", JOptionPane.ERROR_MESSAGE);
            }
        }));
        add(newMenuItem("Polynomial Interactions", (e)->
        {
            String s = JOptionPane.showInputDialog(parent, "Please specify the degree for the transform", "Polynomial Interactions", JOptionPane.QUESTION_MESSAGE);

            try
            {
                int degree = Integer.parseInt(s);
                dataTransformProcess.addTransform(new PolynomialTransform.PolyTransformFactory(degree));
                addTransformName("Polynomial interaction degree " + degree);
            }
            catch (Exception ex)
            {
                //TODO show error
            }
        }));
        add(new JSeparator());
        add(newMenuItem("Clear Transforms", (e) ->
        {
            setDataTransformProcess(new DataTransformProcess());
            currentTransforms.removeAll();
            currentTransforms.add(noTransformsLabel);
        }));
        add(currentTransforms);

    }

    /**
     * Helper method to add the given string name as a item in the list of
     * currently applied transforms. 
     * @param name the name to display to the user
     */
    private void addTransformName(String name)
    {

        if (currentTransforms.getMenuComponent(0) == noTransformsLabel)//no itemns
            currentTransforms.removeAll();
        currentTransforms.add(new JLabel(name));
    }

    /**
     * Helper method to make adding new menu items easier. 
     * @param name the name to display for the item
     * @param action the action to perform
     * @return the JMenuItem with the given name and action
     */
    private static JMenuItem newMenuItem(String name, AbstractActionI action)
    {
        JMenuItem newButton = new JMenuItem(name);
        newButton.setAction(action.getAsAction());
        newButton.setText(name);
        return newButton;
    }

    private void setDataTransformProcess(DataTransformProcess dataTransformProcess)
    {
        this.dataTransformProcess = dataTransformProcess;
    }
    
    /**
     * 
     * @return the current DataTransformProcess that will apply the transforms selected by the user
     */
    public DataTransformProcess getDataTransformProcess()
    {
        return dataTransformProcess;
    }
}

