%For reproducibility
rng(7);
%Viewing Topographic map of Chicago
%viewer = siteviewer("Buildings","chicago.osm","Basemap","topographic");

%Defining base station (transmitter) sites
%%tx properties
tx_names = ["S Clarke", "Monroe", "Adam"];
lats = [41.878313, 41.880680, 41.879405];
lons = [-87.630772, -87.630842, -87.631010];
heights = [6, 6, 6];
freq = 2.4000e9;
transmit_power = 1.0000e-3;
%%Antenna properties
%TODO Change Antenna Properties
antenna_size = [4, 1];
antenna_spacing = [0.0625, 0.0625];
cfgArray = arrayConfig("Size",antenna_size,"ElementSpacing",antenna_spacing);
%%Creating tx
tx = txsite("Name",tx_names,"Latitude",lats,"Longitude",lons,"TransmitterFrequency",freq,"Antenna",cfgArray,"AntennaHeight",heights,"TransmitterPower",transmit_power);
%disp(tx(1,1));
%show(tx);

% %Defining receiver locations from 100GHz data
n_recv = 30000;
load ('temp.mat','lats');
load ('temp.mat','lons');

names = string(1:n_recv);
height = 2;
%%Antenna properties
%TODO Change Antenna Properties
antenna_size = [1, 1];
antenna_spacing = [0.0625, 0.0625];
cfgArray = arrayConfig("Size",antenna_size,"ElementSpacing",antenna_spacing);
rx = rxsite("Name",names,"Latitude",lats,"Longitude",lons,"Antenna",cfgArray,"AntennaHeight",height);

%Ray Tracing Model combined with atmospheric (gas) attenuation
%TODO Change the properties of rptm and raytrace (Building material concrete etc)
rtpm = propagationModel("raytracing",Method="sbr",MaxNumReflections=4);
% gas_pm = propagationModel("gas");
% fog_pm = propagationModel("fog");
rtPlusWeather = rtpm;
%clearMap(viewer);
%raytrace(tx,rx1,rtPlusWeather);
%raytrace(tx,rx2,rtPlusWeather);
%raytrace(tx,rx3,rtPlusWeather);
rays = raytrace(tx,rx,rtPlusWeather);
%rays2 = raytrace(tx,rx2,rtPlusWeather);
%rays3 = raytrace(tx,rx3,rtPlusWeather);

% Deleting old scenario directory and creating new one
if ~exist('./Raytracing_scenarios', 'dir')
    fprintf("Creating Raytracing Scenarios directory\n")
    mkdir './Raytracing_scenarios'
end
if exist('./Raytracing_scenarios/O1_2p4', 'dir')
    fprintf("Deleting old O1_2p4 directory\n")
    rmdir './Raytracing_scenarios/O1_2p4' s
end
fprintf("Creating new O1_2p4 directory\n")
mkdir './Raytracing_scenarios/O1_2p4'

% Storing data for each base station
for bs=1:length(tx_names)
    %%CIR File (Contains phase in degrees, ToA in seconds, and received power in dbm for each path)
    CIR = [n_recv];
    DoD = [n_recv];
    DoA = [n_recv];
    LOS = [n_recv];
    curr_paths = 0;
    curr = 1;
    for i = 1:n_recv
        fprintf('Getting data for base station %i and receiver %i\n',bs,i);
        num_paths_max = 4;
        num_paths = 0;
        CIR = [CIR i 0];
        DoD = [DoD i 0];
        DoA = [DoA i 0];
        curr = curr + 2;
        while num_paths<num_paths_max
    %         fprintf ("%d %d %d %d\n", 1, i, 1, num_paths+1);
    %         fprintf ("%f %f %f\n", rays1{1,i}(1,num_paths+1).PhaseShift, rays1{1,i}(1,num_paths+1).PropagationDelay, rays1{1,i}(1,num_paths+1).PathLoss);
            try
                %LOS Status only for first path
                if num_paths==0
                    LOS_status = rays{bs,i}(1,num_paths+1).LineOfSight;
                    LOS = [LOS LOS_status];
                end
                phase = 180 / pi * rays{bs,i}(1,num_paths+1).PhaseShift;
                ToA = rays{bs,i}(1,num_paths+1).PropagationDelay;
                recv_power = -1 * (rays{bs,i}(1,num_paths+1).PathLoss + 30);
                departure_azi = rays{bs,i}(1,num_paths+1).AngleOfDeparture(1,1);
                departure_ele = rays{bs,i}(1,num_paths+1).AngleOfDeparture(2,1);
                arrival_azi = rays{bs,i}(1,num_paths+1).AngleOfArrival(1,1);
                arrival_ele = rays{bs,i}(1,num_paths+1).AngleOfArrival(2,1);
                num_paths = num_paths + 1;
                CIR = [CIR num_paths phase ToA recv_power];
                DoD = [DoD num_paths departure_azi departure_ele 9999];
                DoA = [DoA num_paths arrival_azi arrival_ele 9999];
                %fprintf ('Happening')
            catch
                %fprintf ('Done for');
                %If broke at 0 paths then put LOS_status = -1
                if num_paths == 0
                    LOS = [LOS -1];
                end
                break
            end
        end
        CIR(1,curr) = num_paths;
        DoD(1,curr) = num_paths;
        DoA(1,curr) = num_paths;
        curr = curr + 4 * num_paths;
    end
    
    %Converting arrays to .mat files for ingestion by deepMIMO generator
    
    save(strcat('./Raytracing_scenarios/O1_2p4/O1_2p4.',int2str(bs),'.CIR.mat'), 'CIR');
    save(strcat('./Raytracing_scenarios/O1_2p4/O1_2p4.',int2str(bs),'.DoD.mat'), 'DoD');
    save(strcat('./Raytracing_scenarios/O1_2p4/O1_2p4.',int2str(bs),'.DoA.mat'), 'DoA');
    save(strcat('./Raytracing_scenarios/O1_2p4/O1_2p4.',int2str(bs),'.LoS.mat'), 'LOS');
end 