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
import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFileChooser;
import jsat.ARFFLoader;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import static jsat.linear.DenseVector.toDenseVec;
import static java.lang.Double.*;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javax.swing.*;
import jsat.*;
import jsat.classifiers.ClassificationDataSet;
import jsat.clustering.*;
import jsat.clustering.dissimilarity.AverageLinkDissimilarity;
import jsat.clustering.dissimilarity.CentroidDissimilarity;
import jsat.clustering.dissimilarity.CompleteLinkDissimilarity;
import jsat.clustering.dissimilarity.SingleLinkDissimilarity;
import jsat.clustering.dissimilarity.WardsDissimilarity;
import jsat.clustering.evaluation.DunnIndex;
import jsat.clustering.evaluation.intra.MeanCentroidDistance;
import jsat.clustering.hierarchical.DivisiveGlobalClusterer;
import jsat.clustering.hierarchical.DivisiveLocalClusterer;
import jsat.clustering.hierarchical.PriorityHAC;
import jsat.clustering.kmeans.*;
import jsat.datatransform.LinearTransform;
import jsat.guitool.ParameterPanel;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.NormalizedEuclideanDistance;
import jsat.linear.vectorcollection.VectorArray;
import jsat.parameters.Parameterized;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
@SuppressWarnings("serial")
public class ClusterToyWorld extends javax.swing.JFrame
{
    private static DataSet dataSet;
    final JFileChooser fileChooser = new JFileChooser();
    private static JTabbedPane centerTabbed;
    private static TransformsMenu transformsMenu;
    private static final AtomicInteger waitingFor = new AtomicInteger(0);
    private static BlockingQueue<Runnable> backgroundJobQueue;
    private static Thread backgroundThread;
    private static final ExecutorService execService = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    
    private static final Map<String, Clusterer> clusterers = new LinkedHashMap<String, Clusterer>()
    {{
        put("ElkanKMeans", new ElkanKMeans());
        put("EMGaussianMixture", new EMGaussianMixture());
        put("MeanShift", new MeanShift());
        put("CLARA", new CLARA());
        put("DBSCAN", new DBSCAN(new NormalizedEuclideanDistance(), new VectorArray.VectorArrayFactory()));
        put("LSDBC", new LSDBC());
        put("OPTICS", new OPTICS());
        put("HAC : Single Link", new PriorityHAC(new SingleLinkDissimilarity(new EuclideanDistance())));
        put("HAC : Complete Link", new PriorityHAC(new CompleteLinkDissimilarity(new EuclideanDistance())));
        put("HAC : Average Link", new PriorityHAC(new AverageLinkDissimilarity(new EuclideanDistance())));
        put("HAC : Centroid Link", new PriorityHAC(new CentroidDissimilarity(new EuclideanDistance())));
        put("HAC : Ward", new PriorityHAC(new WardsDissimilarity()));
        put("DivisiveLocalClusterer", new DivisiveLocalClusterer(new ElkanKMeans(), new DunnIndex(new MeanCentroidDistance(), new AverageLinkDissimilarity())));
        put("DivisiveGlobalClusterer", new DivisiveGlobalClusterer(new ElkanKMeans(), new DunnIndex(new MeanCentroidDistance(), new AverageLinkDissimilarity())));
        put("FLAME", new FLAME(new EuclideanDistance(), 50, 5000));
        
    }};
    
    private void addClusteringToTabbedDisplay(int kSize, int[] assignments, final String fullName)
    {
        int min = 0;
        for(int i : assignments)
            min = Math.min(i, min);
        if(min < 0)
            kSize++;
        CategoricalData cd = new CategoricalData(kSize);
        if(min < 0)
            cd.setOptionName("Noise", kSize-1);
        ClassificationDataSet clustering = new ClassificationDataSet(2, new CategoricalData[0], cd);
        int[] noCats = new int[0];
        for (int i = 0; i < assignments.length; i++)
        {
            if (assignments[i] >= 0)
                clustering.addDataPoint(dataSet.getDataPoint(i).getNumericalValues(), noCats, assignments[i]);
            else
                clustering.addDataPoint(dataSet.getDataPoint(i).getNumericalValues(), noCats, kSize-1);
        }

        
        SwingUtilities.invokeLater(() ->
        {
            final JFXPanel fxPanel = new JFXPanel();
            Platform.runLater(() ->
            {
                fxPanel.setScene(new Scene(new BorderPane(Plot.scatterC(clustering))));
            });
            centerTabbed.add(fullName, fxPanel);
            centerTabbed.setSelectedIndex(centerTabbed.getTabCount() - 1);
            getContentPane().validate();
            getContentPane().repaint();
        });
        
    }

    /**
     * Creates new form ClusterToyWorld
     */
    public ClusterToyWorld()
    {
        initComponents();
        jMenuBar1.add(transformsMenu = new TransformsMenu(this));
        jLabel1.setText(" ");
        backgroundJobQueue = new LinkedBlockingQueue<>();
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
                        jLabel1.setText("Waiting on " + now + " jobs...");
                    else
                        jLabel1.setText(" ");
                }
                catch (InterruptedException ex)
                {
                    Logger.getLogger(ClusterToyWorld.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        });
        backgroundThread.start();
        jLabel1.setFont(jLabel1.getFont().deriveFont(16));

        
        Map<String, JMenu> autoClusterSubMenu = new HashMap<>();
        Map<String, JMenu> kClusterSubMenu = new HashMap<>();
        for(Entry<String, Clusterer> entry : clusterers.entrySet())
        {
            final Clusterer clusterer = entry.getValue();
            final String clusterName = entry.getKey();
            {
                JMenu targetMenu;
                final String menuName;
                if(clusterName.contains(":"))
                {
                    String[] split = clusterName.split(":");
                    menuName = split[1].trim();
                    if(!autoClusterSubMenu.containsKey(split[0]))
                    {
                        JMenu jMenu = new JMenu(split[0].trim());
                        autoClusterSubMenu.put(split[0], jMenu );
                        jMenuAutoCluster.add(jMenu);
                    }
                    targetMenu = autoClusterSubMenu.get(split[0]);
                }
                else
                {
                    targetMenu = jMenuAutoCluster;
                    menuName = clusterName;
                }
                
                JMenuItem clusterItem = new JMenuItem(menuName);
                clusterItem.addActionListener((ActionEvent ae) ->
                {
                    int now = waitingFor.incrementAndGet();
                    jLabel1.setText("Waiting on " + now + " jobs...");
                    try
                    {
                        if(clusterer instanceof Parameterized)
                        {
                            ParameterPanel pp = new ParameterPanel((Parameterized) clusterer);
                            final JDialog jd = new JDialog(getOwner(), "Set Parameters", Dialog.ModalityType.APPLICATION_MODAL);
                            jd.setContentPane(pp);
                            pp.getjButtonOk().addActionListener((ActionEvent e) ->
                            {
                                jd.setVisible(false);
                            });
                            jd.pack();
                            jd.setVisible(true);
                        }
                        backgroundJobQueue.put((Runnable) () ->
                        {
                            int[] assignments = clusterer.cluster(dataSet, execService, (int[]) null);
                            String fullName = "Auto " + menuName;
                            int kSize = 0;
                            for (int i1 : assignments)
                                kSize = Math.max(kSize, i1 + 1);
                            addClusteringToTabbedDisplay(kSize, assignments, fullName);
                        });
                    }
                    catch (InterruptedException ex)
                    {
                        Logger.getLogger(ClusterToyWorld.class.getName()).log(Level.SEVERE, null, ex);
                    }
                });

                
                targetMenu.add(clusterItem);
            }
            
            if(clusterer instanceof KClusterer)
            {
                JMenu targetMenu;
                final String menuName;
                if(clusterName.contains(":"))
                {
                    String[] split = clusterName.split(":");
                    menuName = split[1].trim();
                    if(!kClusterSubMenu.containsKey(split[0]))
                    {
                        JMenu jMenu = new JMenu(split[0].trim());
                        kClusterSubMenu.put(split[0], jMenu );
                        jMenuKCluster.add(jMenu);
                    }
                    targetMenu = kClusterSubMenu.get(split[0]);
                }
                else
                {
                    targetMenu = jMenuKCluster;
                    menuName = clusterName;
                }
                
                JMenuItem clusterItem = new JMenuItem(menuName);
                clusterItem.addActionListener((ActionEvent ae) ->
                {
                    if(clusterer instanceof Parameterized)
                    {
                        ParameterPanel pp = new ParameterPanel((Parameterized) clusterer);
                        final JDialog jd = new JDialog(getOwner(), "Set Parameters", Dialog.ModalityType.APPLICATION_MODAL);
                        jd.setContentPane(pp);
                        pp.getjButtonOk().addActionListener((ActionEvent e) ->
                        {
                            jd.setVisible(false);
                        });
                        jd.pack();
                        jd.setVisible(true);
                    }
                    String value = JOptionPane.showInputDialog("Please specify the number of clusters");
                    if(value == null)
                        return;
                    final int kSize = Integer.parseInt(value);
                    if(kSize < 0)
                        return;//TODO show an error dialog
                    
                    int now = waitingFor.incrementAndGet();
                    jLabel1.setText("Waiting on " + now + " jobs...");
                    try
                    {
                        backgroundJobQueue.put((Runnable) () ->
                        {
                            int[] assignments = ((KClusterer)clusterer).cluster(dataSet, kSize, execService, (int[])null);
                            String fullName = "k = " + kSize + " " + menuName;
                            addClusteringToTabbedDisplay(kSize, assignments, fullName);
                        });
                    }
                    catch (InterruptedException ex)
                    {
                        Logger.getLogger(ClusterToyWorld.class.getName()).log(Level.SEVERE, null, ex);
                    }
                });

                targetMenu.add(clusterItem);
            
            }
        }
        
        setSize(600, 400);
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

        jPanel1 = new javax.swing.JPanel();
        jLabel1 = new javax.swing.JLabel();
        jMenuBar1 = new javax.swing.JMenuBar();
        jMenuFile = new javax.swing.JMenu();
        jMenuItemOpen = new javax.swing.JMenuItem();
        jMenuItemOpenTxt = new javax.swing.JMenuItem();
        jMenuAutoCluster = new javax.swing.JMenu();
        jMenuKCluster = new javax.swing.JMenu();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jPanel1.setBorder(javax.swing.BorderFactory.createEtchedBorder());
        jPanel1.setLayout(new java.awt.FlowLayout(java.awt.FlowLayout.RIGHT));

        jLabel1.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        jLabel1.setText("This text will change");
        jPanel1.add(jLabel1);

        getContentPane().add(jPanel1, java.awt.BorderLayout.PAGE_END);

        jMenuFile.setText("File");

        jMenuItemOpen.setText("Open");
        jMenuItemOpen.addActionListener(new java.awt.event.ActionListener()
        {
            public void actionPerformed(java.awt.event.ActionEvent evt)
            {
                jMenuItemOpenActionPerformed(evt);
            }
        });
        jMenuFile.add(jMenuItemOpen);
        jMenuFile.add(jMenuItemOpenTxt);

        jMenuBar1.add(jMenuFile);

        jMenuAutoCluster.setText("Cluster");
        jMenuBar1.add(jMenuAutoCluster);

        jMenuKCluster.setText("K-Cluster");
        jMenuBar1.add(jMenuKCluster);

        setJMenuBar(jMenuBar1);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jMenuItemOpenActionPerformed(java.awt.event.ActionEvent evt)//GEN-FIRST:event_jMenuItemOpenActionPerformed
    {//GEN-HEADEREND:event_jMenuItemOpenActionPerformed
        int returnVal = fileChooser.showOpenDialog(this);
        if (returnVal == JFileChooser.APPROVE_OPTION) 
        {
            File file = fileChooser.getSelectedFile();
            String extension = file.toString().substring(file.toString().lastIndexOf(".")+1);

            DataSet tmpDataSet = null;
            
            if(extension.equalsIgnoreCase("arff"))
            {
                tmpDataSet = ARFFLoader.loadArffFile(file);
            }
            else if(extension.equalsIgnoreCase("txt"))
            {
                List<DataPoint> dpList = new ArrayList<>();
                int[] noCatVals = new int[0];
                CategoricalData[] noCats = new CategoricalData[0];
                try
                {
                    BufferedReader br = new BufferedReader(new FileReader(file));
                    String line;
                    while((line = br.readLine()) != null)
                    {
                        String[] split = line.trim().split("[\\s,]+");
                        if(split.length < 2)
                            continue;
                        DataPoint dp = new DataPoint(toDenseVec(parseDouble(split[0]),parseDouble(split[1])), noCatVals, noCats);
                        dpList.add(dp);
                    }
                    
                    tmpDataSet = new SimpleDataSet(dpList);
                }
                catch(IOException ex)
                {
                    
                }
            }
            
            if(tmpDataSet.getNumNumericalVars() != 2)
                return;//TODO throw an error
            
            dataSet = tmpDataSet;
            dataSet.applyTransform(new LinearTransform(dataSet, 0, 1));
            
            final JFXPanel fxPanel = new JFXPanel();
            Platform.runLater(() ->
            {
                fxPanel.setScene(new Scene(new BorderPane(Plot.scatter(dataSet))));
            });
            if(centerTabbed != null)
                remove(centerTabbed);
            centerTabbed = new JTabbedPane();
            centerTabbed.add("Original Data Set", fxPanel);
            add(centerTabbed, BorderLayout.CENTER);
            getContentPane().validate();
            getContentPane().repaint();
        }
    }//GEN-LAST:event_jMenuItemOpenActionPerformed

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
            new ClusterToyWorld().setVisible(true);
        });
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel jLabel1;
    private javax.swing.JMenu jMenuAutoCluster;
    private javax.swing.JMenuBar jMenuBar1;
    private javax.swing.JMenu jMenuFile;
    private javax.swing.JMenuItem jMenuItemOpen;
    private javax.swing.JMenuItem jMenuItemOpenTxt;
    private javax.swing.JMenu jMenuKCluster;
    private javax.swing.JPanel jPanel1;
    // End of variables declaration//GEN-END:variables
}
