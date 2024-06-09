#define CONFIG_CATCH_MAIN
#include <catch2/catch_all.hpp>

int main(int argc, char *argv[])
{
    // Adjust the verbosity level
    Catch::Session session;

    // Set up the configData
    Catch::ConfigData configData;
    configData.verbosity = Catch::Verbosity::High; // Set verbosity to high
    session.configData().verbosity = Catch::Verbosity::High;

    // Run the tests
    int returnCode = session.run(argc, argv);

    return returnCode;
}