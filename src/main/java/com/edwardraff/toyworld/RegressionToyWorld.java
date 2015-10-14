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

import com.edwardraff.jsatfx.Plot;
import com.edwardraff.jsatfx.swing.ParameterPanel;
import static java.lang.Math.*;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javax.swing.*;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.classifiers.boosting.Bagging;
import jsat.classifiers.knn.LWL;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.classifiers.svm.DCDs;
import jsat.classifiers.trees.*;
import jsat.datatransform.DataModelPipeline;
import jsat.distributions.*;
import jsat.distributions.empirical.kernelfunc.EpanechnikovKF;
import jsat.distributions.kernels.RBFKernel;
import jsat.distributions.multivariate.MetricKDE;
import jsat.linear.DenseVector;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.DefaultVectorCollectionFactory;
import jsat.math.Function;
import jsat.parameters.DoubleParameter;
import jsat.parameters.IntParameter;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.*;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class RegressionToyWorld extends javax.swing.JFrame
{
    /**
     * The main holder of all the visualizations. Empty until a dataset is 
     * loaded. The first index will be a visualization of the dataset, all 
     * subsequent indices are regressors we have trained 
     */
    private JTabbedPane jTabbedPane;
    /**
     * This will be the currently loaded dataset
     */
    private static RegressionDataSet rData;
    /**
     * The number of data points to generate evenly along the range for every 
     * pass through the data
     */
    private int genSize = 700;
    /**
     * The multiple applied to {@link #genSize} for how many complete noise data
     * points to generate (and are only generated when the random noise 
     * selection is set)
     */
    private double randNoieFrac = 0.2;
    /**
     * The number of passes through the range to generate data for
     */
    private int passes = 1;
    /**
     * The starting value on the x axis to generate data from
     */
    private static double genStart = -4;
    /**
     * The ending value on the x axis to generate data from
     */
    private static double genEnd = 4;
    
    /**
     * Menu to hold all the transformation options
     */
    private static TransformsMenu transformsMenu;
    
    /**
     * Exploiting the parameterized code in JSAT to make a lazy GUI to configure 
     * options when generating test datasets
     */
    final List<Parameter> appParams = new ArrayList<Parameter>()
    {{
        add(new IntParameter() {

            @Override
            public int getValue()
            {
                return genSize;
            }

            @Override
            public boolean setValue(int val)
            {
                if(val <= 0)
                    return false;
                genSize = val;
                return true;
            }

            @Override
            public String getASCIIName()
            {
                return "Sample Size";
            }
        });
        
        add(new DoubleParameter() {

            @Override
            public double getValue()
            {
                return genStart;
            }

            @Override
            public boolean setValue(double val)
            {
                genStart = val;
                return true;
            }

            @Override
            public String getASCIIName()
            {
                return "Gen Start";
            }
        });
        
        add(new DoubleParameter() {

            @Override
            public double getValue()
            {
                return genEnd;
            }

            @Override
            public boolean setValue(double val)
            {
                genEnd = val;
                return true;
            }

            @Override
            public String getASCIIName()
            {
                return "Gen End";
            }
        });
        
        add(new DoubleParameter() {

            @Override
            public double getValue()
            {
                return randNoieFrac;
            }

            @Override
            public boolean setValue(double val)
            {
                if(val <= 0 || val > 1)
                    return false;
                randNoieFrac = val;
                return true;
            }

            @Override
            public String getASCIIName()
            {
                return "Random Noise Fraction";
            }
        });
    }};
    
    /**
     * Atomic integer keeps track of the number of regressors we are currently 
     * waiting to finish training
     */
    private static final AtomicInteger waitingFor = new AtomicInteger(0);
    /**
     * Queue of classification jobs to run
     */
    private static BlockingQueue<Runnable> backgroundJobQueue;
    /**
     * The thread we use to eat classification jobs forever
     */
    private static Thread backgroundThread;
    
    private static final ExecutorService execService = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    
    /**
     * Map of all the regressors we will be using
     */
    private static final Map<String, Regressor> regressors = new LinkedHashMap<String, Regressor>()
    {{
        put("k-NearestNeighbour", new NearestNeighbour(1));
        put("Locally Weighted Linear Regression", new LWL(new MultipleLinearRegression(true), 15, new EuclideanDistance(), 
            EpanechnikovKF.getInstance(), new DefaultVectorCollectionFactory()));
        put("NadarayaWatson", new NadarayaWatson(new MetricKDE()));
        put("Linear Regression", new MultipleLinearRegression());
        put("Linear SVR", new DCDs());
        put("RidgeRegression", new RidgeRegression(0.01));
        put("Decision Stump", new DecisionStump());
        put("Decision Tree", new DecisionTree());
        put("RandomForest", new RandomForest(100));
        put("SGB-Stumps", new StochasticGradientBoosting(new DecisionStump(), 200));
        put("SGB-Trees", new StochasticGradientBoosting(new DecisionTree(3, 10, TreePruner.PruningMethod.NONE, 0.1), 200));
        put("KernelRLS", new KernelRLS(new RBFKernel(0.075), 0.001));
        
    }};
    
    /**
     * This object holds the 1D function that represents the grown truth of what
     * we are trying to approximate. If {@code null} no errors will be shown. If
     * non-null all plots will draw the true function we are trying to learn
     */
    static private Function truth = null;
    
    /**
     * All the functions we can generate data for
     */
    private static final Map<String, Func1D> generatableFunctions = new LinkedHashMap<String, Func1D>()
    {{
        put("x*2.5", (x) -> x*2.5);
        put("sin(x)", (x) -> sin(x));
        put("sin(x)*(x^2+x)", (x) -> sin(x)*(x*x+x));
        put("exp(x)", (x) -> exp(x));
        put("exp(sin(x))+x/3", (x) -> exp(sin(x))+x/3);
        put("exp(sin(x))+cos(x)*sign(sin(x*min(x+1,3)^2))", (x) -> exp(sin(x))+cos(x)*Math.signum(sin(x*pow(min(x+1,3),2))));
    }};
    

    /**
     * Creates new form RegressionToyWorld
     */
    public RegressionToyWorld()
    {
        initComponents();
        jMenuBar1.add(transformsMenu = new TransformsMenu(this, "Transforms"));
        jLabelInfo.setText(" ");
        backgroundJobQueue = new LinkedBlockingQueue<>();
        
        for(Entry<String, Regressor> entry : regressors.entrySet())
        {
            final String name = entry.getKey();
            final Regressor regressorToClone = entry.getValue();
            JMenuItem jitem = new JMenuItem(name);
            jitem.addActionListener((ActionEvent e) ->
            {
                try
                {
                    Regressor regressor = regressorToClone.clone();
                    
                    ClassifierToyWorld.showParameterizedDialog(this, regressor);
                    
                    if(jRadioButtonMetaRANSAC.isSelected())
                    {
                        RANSAC ransac = new RANSAC(regressor, 100, 20, 50, 10);
                        ParameterPanel.showParameterDiag(getOwner(), "Set RANSAC Parameters", ransac);
                        
                        regressor = ransac;
                    }
                    else if(jRadioButtonMetaBagging.isSelected())
                    {
                        Bagging bagging = new Bagging(regressor);
                        ParameterPanel.showParameterDiag(getOwner(), "Set Bagging Parameters", bagging);
                        
                        regressor = bagging;
                    }
                    
                    ///Create tranformed version
                    regressor = new DataModelPipeline(regressor, transformsMenu.getDataTransformProcess().clone());
                    final Regressor regressorToUse = regressor.clone();
                    
                    jLabelInfo.setText("Waiting for " + waitingFor.incrementAndGet() + " jobs to finish");
                    backgroundJobQueue.put((Runnable) () ->
                    {
                        regressorToUse.train(rData, execService);

                        final JFXPanel fxPanel = new JFXPanel();
                        Platform.runLater(() ->
                        {
                            fxPanel.setScene(new Scene(new BorderPane(Plot.regression(rData, regressorToUse)  )));
                        });
                        jTabbedPane.add(name, fxPanel);
                    });
                }
                catch (InterruptedException ex)
                {
                    Logger.getLogger(RegressionToyWorld.class.getName()).log(Level.SEVERE, null, ex);
                }
                
            });
            jMenuRegression.add(jitem);
        }
        
        
        ///Generators
        
        for(final Entry<String, Func1D> entry : generatableFunctions.entrySet())
        {
            JMenuItem jitem = new JMenuItem(entry.getKey());
            
            jitem.addActionListener((ActionEvent e) ->
            {
                truth = entry.getValue();
                rData = new RegressionDataSet(1, new CategoricalData[0]);
                Distribution outNoise = getOutputNoise();
                Distribution inNoise = getInputNoise();
                Random rand = new Random();
                for (int pass = 0; pass < passes; pass++)
                    for (double x1 = genStart; x1 < genEnd; x1 += (genEnd-genStart)/genSize)
                        addPoint(x1 + inNoise.invCdf(rand.nextDouble()), entry.getValue().f(x1) + outNoise.invCdf(rand.nextDouble()));
                if(!jRBRandNoNoise.isSelected())
                {
                    //Only option is unfirom ATM
                    Distribution xDist = new Uniform(genStart, genEnd);
                    Distribution yDist = new Uniform(rData.getTargetValues().min(), rData.getTargetValues().max());
                    
                    for(int i = 0; i < genSize*randNoieFrac; i++)
                        addPoint(xDist.invCdf(rand.nextDouble()), yDist.invCdf(rand.nextDouble()));
                    
                }
                setUpMain();
            });
            jMenuGenerateData.add(jitem);
        }
        
        //loop forever, eating jobs and updating the counter
        backgroundThread = new Thread(() -> 
        {
            while(true)
            {
                try
                {
                    Runnable toRun = backgroundJobQueue.take();
                    validate();
                    repaint();
                    toRun.run();
                    int now = waitingFor.decrementAndGet();
                    if(now > 0)
                        jLabelInfo.setText("Waiting on " + now + " jobs...");
                    else
                        jLabelInfo.setText(" ");
                }
                catch (InterruptedException ex)
                {
                    Logger.getLogger(ClusterToyWorld.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        });
        backgroundThread.setDaemon(true);
        backgroundThread.start();
        
        setSize(600, 400);
    }

    /**
     * Adds a data point to the current data set
     * @param x the value of the input
     * @param y the value of the response
     */
    private void addPoint(double x, double y)
    {
        rData.addDataPoint(new DataPoint(DenseVector.toDenseVec(x), new int[0], new CategoricalData[0]), y);
    }
    
    private Distribution getOutputNoise()
    {
        if(jRBOutUniformNoise.isSelected())
            return new Uniform(-0.2, 0.2);
        else if(jRBOutGaussNoise.isSelected())
            return new Normal(0, 0.2);
        return new Uniform(0, Math.nextUp(0));
    }
    
    private Distribution getInputNoise()
    {
        if(jRBInUniformNoise.isSelected())
            return new Uniform(-0.2, 0.2);
        else if(jRBInGaussNoise.isSelected())
            return new Normal(0, 0.2);
        return new Uniform(0, Math.nextUp(0));
    }
    
    
    private void setUpMain()
    {
        if (jTabbedPane != null)
            remove(jTabbedPane);
        jTabbedPane = new JTabbedPane();
        
        final JFXPanel fxPanel = new JFXPanel();
        Platform.runLater(() ->
        {
            fxPanel.setScene(new Scene(new BorderPane(Plot.regression(rData, (double value) -> truth.f(value)))));
        });
        
        jTabbedPane.add("Raw Data", fxPanel);
        add(jTabbedPane, BorderLayout.CENTER);
        repaint();
        validate();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents()
    {

        buttonGroupGenOutNoise = new ButtonGroup();
        buttonGroupGenInNoise = new ButtonGroup();
        buttonGroupGenRandomNoise = new ButtonGroup();
        buttonGroupMeta = new ButtonGroup();
        bottomAreaPanel = new JPanel();
        jLabelInfo = new JLabel();
        jMenuBar1 = new JMenuBar();
        jMenu1 = new JMenu();
        jMenuGenerateData = new JMenu();
        jMenuItemGenParams = new JMenuItem();
        jMenuRandNoise = new JMenu();
        jRBRandNoNoise = new JRadioButtonMenuItem();
        jRBRandUniformNoise = new JRadioButtonMenuItem();
        jMenuOutNoise = new JMenu();
        jRBOutNoNoise = new JRadioButtonMenuItem();
        jRBOutUniformNoise = new JRadioButtonMenuItem();
        jRBOutGaussNoise = new JRadioButtonMenuItem();
        jMenuInNoise = new JMenu();
        jRBInNoNoise = new JRadioButtonMenuItem();
        jRBInUniformNoise = new JRadioButtonMenuItem();
        jRBInGaussNoise = new JRadioButtonMenuItem();
        jSeparator1 = new JPopupMenu.Separator();
        jMenuRegression = new JMenu();
        jMenuMeta = new JMenu();
        jRadioButtonMetaNone = new JRadioButtonMenuItem();
        jRadioButtonMetaRANSAC = new JRadioButtonMenuItem();
        jRadioButtonMetaBagging = new JRadioButtonMenuItem();

        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        bottomAreaPanel.setBorder(BorderFactory.createEtchedBorder());
        bottomAreaPanel.setLayout(new FlowLayout(FlowLayout.RIGHT));

        jLabelInfo.setHorizontalAlignment(SwingConstants.CENTER);
        jLabelInfo.setText("This text will change");
        bottomAreaPanel.add(jLabelInfo);

        getContentPane().add(bottomAreaPanel, BorderLayout.PAGE_END);

        jMenu1.setText("File");

        jMenuGenerateData.setText("Generate");

        jMenuItemGenParams.setText("Parameters");
        jMenuItemGenParams.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent evt)
            {
                jMenuItemGenParamsActionPerformed(evt);
            }
        });
        jMenuGenerateData.add(jMenuItemGenParams);

        jMenuRandNoise.setText("Random Noise");

        buttonGroupGenRandomNoise.add(jRBRandNoNoise);
        jRBRandNoNoise.setSelected(true);
        jRBRandNoNoise.setText("No");
        jMenuRandNoise.add(jRBRandNoNoise);

        buttonGroupGenRandomNoise.add(jRBRandUniformNoise);
        jRBRandUniformNoise.setText("Uniform");
        jMenuRandNoise.add(jRBRandUniformNoise);

        jMenuGenerateData.add(jMenuRandNoise);

        jMenuOutNoise.setText("Output Noise");

        buttonGroupGenOutNoise.add(jRBOutNoNoise);
        jRBOutNoNoise.setText("None");
        jMenuOutNoise.add(jRBOutNoNoise);

        buttonGroupGenOutNoise.add(jRBOutUniformNoise);
        jRBOutUniformNoise.setText("Uniform");
        jMenuOutNoise.add(jRBOutUniformNoise);

        buttonGroupGenOutNoise.add(jRBOutGaussNoise);
        jRBOutGaussNoise.setSelected(true);
        jRBOutGaussNoise.setText("Gaussian");
        jMenuOutNoise.add(jRBOutGaussNoise);

        jMenuGenerateData.add(jMenuOutNoise);

        jMenuInNoise.setText("Input noise");

        buttonGroupGenInNoise.add(jRBInNoNoise);
        jRBInNoNoise.setSelected(true);
        jRBInNoNoise.setText("None");
        jMenuInNoise.add(jRBInNoNoise);

        buttonGroupGenInNoise.add(jRBInUniformNoise);
        jRBInUniformNoise.setText("Uniform");
        jMenuInNoise.add(jRBInUniformNoise);

        buttonGroupGenInNoise.add(jRBInGaussNoise);
        jRBInGaussNoise.setText("Gaussian");
        jMenuInNoise.add(jRBInGaussNoise);

        jMenuGenerateData.add(jMenuInNoise);
        jMenuGenerateData.add(jSeparator1);

        jMenu1.add(jMenuGenerateData);

        jMenuBar1.add(jMenu1);

        jMenuRegression.setText("Regression");
        jMenuBar1.add(jMenuRegression);

        jMenuMeta.setText("Meta");

        buttonGroupMeta.add(jRadioButtonMetaNone);
        jRadioButtonMetaNone.setSelected(true);
        jRadioButtonMetaNone.setText("None");
        jMenuMeta.add(jRadioButtonMetaNone);

        buttonGroupMeta.add(jRadioButtonMetaRANSAC);
        jRadioButtonMetaRANSAC.setText("RANSAC");
        jMenuMeta.add(jRadioButtonMetaRANSAC);

        buttonGroupMeta.add(jRadioButtonMetaBagging);
        jRadioButtonMetaBagging.setText("Bagging");
        jMenuMeta.add(jRadioButtonMetaBagging);

        jMenuBar1.add(jMenuMeta);

        setJMenuBar(jMenuBar1);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jMenuItemGenParamsActionPerformed(ActionEvent evt)//GEN-FIRST:event_jMenuItemGenParamsActionPerformed
    {//GEN-HEADEREND:event_jMenuItemGenParamsActionPerformed
        ParameterPanel.showParameterDiag(getOwner(), "Generation parameters", new Parameterized() {

            @Override
            public List<Parameter> getParameters()
            {
                return appParams;
            }

            @Override
            public Parameter getParameter(String paramName)
            {
                return Parameter.toParameterMap(appParams).get(paramName);
            }
        });
    }//GEN-LAST:event_jMenuItemGenParamsActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[])
    {
        //For OSX, dosn't impact anyone else - so who cares. 
        System.setProperty("apple.laf.useScreenMenuBar", "true");

        /*
         * Create and display the form
         */
        java.awt.EventQueue.invokeLater(() ->
        {
            new RegressionToyWorld().setVisible(true);
        });
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private JPanel bottomAreaPanel;
    private ButtonGroup buttonGroupGenInNoise;
    private ButtonGroup buttonGroupGenOutNoise;
    private ButtonGroup buttonGroupGenRandomNoise;
    private ButtonGroup buttonGroupMeta;
    private JLabel jLabelInfo;
    private JMenu jMenu1;
    private JMenuBar jMenuBar1;
    private JMenu jMenuGenerateData;
    private JMenu jMenuInNoise;
    private JMenuItem jMenuItemGenParams;
    private JMenu jMenuMeta;
    private JMenu jMenuOutNoise;
    private JMenu jMenuRandNoise;
    private JMenu jMenuRegression;
    private JRadioButtonMenuItem jRBInGaussNoise;
    private JRadioButtonMenuItem jRBInNoNoise;
    private JRadioButtonMenuItem jRBInUniformNoise;
    private JRadioButtonMenuItem jRBOutGaussNoise;
    private JRadioButtonMenuItem jRBOutNoNoise;
    private JRadioButtonMenuItem jRBOutUniformNoise;
    private JRadioButtonMenuItem jRBRandNoNoise;
    private JRadioButtonMenuItem jRBRandUniformNoise;
    private JRadioButtonMenuItem jRadioButtonMetaBagging;
    private JRadioButtonMenuItem jRadioButtonMetaNone;
    private JRadioButtonMenuItem jRadioButtonMetaRANSAC;
    private JPopupMenu.Separator jSeparator1;
    // End of variables declaration//GEN-END:variables
}
