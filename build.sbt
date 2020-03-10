name := "GauGen"

version := "0.1"

scalaVersion := "2.12.8"

libraryDependencies ++= Seq("org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2")

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
resolvers ++= Seq(Resolver.sonatypeRepo("snapshots"))
