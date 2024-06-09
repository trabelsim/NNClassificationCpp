// catch_config.hpp

#pragma once

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

namespace Catch
{
    struct ConfigData;
    struct Session;
}

extern Catch::ConfigData &getMyConfigData(); // Declaration of a function to get the config data
extern Catch::Session &getMySession();       // Declaration of a function to get the Catch session

// Define the configuration settings
Catch::ConfigData &getMyConfigData()
{
    static Catch::ConfigData configData;
    configData.verbosity = Catch::Verbosity::High; // Set verbosity level as needed
    // Add more configuration settings here
    return configData;
}

// Define the Catch session
Catch::Session &getMySession()
{
    static Catch::Session session;
    return session;
}
