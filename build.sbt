name := """SMS classifier"""

version := "1.0-SNAPSHOT"

lazy val root = (project in file(".")).enablePlugins(PlayJava)

scalaVersion := "2.12.2"

libraryDependencies += guice

libraryDependencies += "nz.ac.waikato.cms.weka" % "weka-stable" % "3.8.0"